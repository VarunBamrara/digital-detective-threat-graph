"""
Digital Detective — Data Acquisition
=====================================
Downloads publicly available APT threat-intelligence PDF reports from
the APTnotes project (https://github.com/kbandla/APTnotes) into
digital_detective_data/aptnotes_pdfs/ for Phase 1 to process.

IMPORTANT — please read:
--------------------------
The APTnotes maintainers ran out of GitHub storage years ago. As a
result:
  - Only a small "historical" set (2008–2011, ~8 reports) is still
    hosted directly in the git repo and can be downloaded reliably
    with a plain HTTP request. This script always fetches those.
  - Everything from 2012 onward (~1000+ reports) is now indexed in
    APTnotes_summary.csv but the actual PDFs live on Box.com as
    individual "shared link" pages, not as direct downloadable files.
    Box shared links usually require a browser/JS session to resolve,
    so bulk-downloading them with a plain script is unreliable. This
    script makes a best-effort attempt at a limited number of them
    (using the documented "/download" suffix that Box sometimes
    honors for public links) and clearly reports which ones failed.
    Failures here are expected and do not break the pipeline.

If you want a larger corpus than what this script reliably gets you,
the most dependable option is to open a handful of links from
digital_detective_data/APTnotes_summary.csv in a browser yourself and
drop the downloaded PDFs into digital_detective_data/aptnotes_pdfs/ —
Phase 1 will pick up anything placed there.

Usage:
    pip install requests tqdm
    python download.py
"""

import csv
import io
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("digital_detective_data")
PDF_DIR      = DATA_DIR / "aptnotes_pdfs"
INDEX_CSV    = DATA_DIR / "APTnotes_summary.csv"

RAW_BASE     = "https://raw.githubusercontent.com/kbandla/APTnotes/master/historical"
CSV_URL      = "https://raw.githubusercontent.com/kbandla/APTnotes/master/APTnotes_summary.csv"

# The "historical" folder is the only part of the repo where actual PDF
# blobs still exist (rest were migrated to Box). Paths taken directly
# from the repo tree.
HISTORICAL_PDFS = [
    "2008/Cyberwar.pdf",
    "2008/chinas-electronic.pdf",
    "2009/Ashmore - Impact of Alleged Russian Cyber Attacks .pdf",
    "2009/Cyber-030.pdf",
    "2009/DECLAWING THE DRAGON.pdf",
    "2011/CyberEspionage.pdf",
    "2011/enter-the-cyberdragon.pdf",
    "2011/vol7no2Ball.pdf",
]

# How many additional Box-hosted reports to attempt (best-effort only).
# Kept modest to avoid hammering Box and because the success rate is
# inherently low for a plain script.
MAX_BOX_ATTEMPTS = 40
BOX_TIMEOUT      = 12
REQUEST_DELAY    = 1.0

HEADERS = {"User-Agent": "Mozilla/5.0 (DigitalDetective research script)"}


# ── Step 1: guaranteed downloads from the git-hosted "historical" set ────────

def download_historical(pdf_dir: Path) -> int:
    print("  Step 1 — Downloading git-hosted historical reports (reliable)")
    ok = 0
    for rel_path in tqdm(HISTORICAL_PDFS, desc="  Historical", ncols=80, unit="pdf"):
        url = f"{RAW_BASE}/{requests.utils.quote(rel_path)}"
        out_name = Path(rel_path).name
        out_path = pdf_dir / out_name

        if out_path.exists():
            ok += 1
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
            ok += 1
        except Exception as e:
            tqdm.write(f"    ✗ {out_name} — {e}")

    print(f"  Historical reports saved : {ok} / {len(HISTORICAL_PDFS)}\n")
    return ok


# ── Step 2: fetch the full metadata index (Filename/Title/Source/Link/Year) ──

def download_index(index_csv: Path) -> list[dict]:
    print("  Step 2 — Downloading full report index (metadata for ~1000+ reports)")
    try:
        resp = requests.get(CSV_URL, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        index_csv.parent.mkdir(parents=True, exist_ok=True)
        index_csv.write_bytes(resp.content)

        rows = list(csv.DictReader(io.StringIO(resp.content.decode("utf-8"))))
        print(f"  Index saved              : {index_csv}  ({len(rows)} reports listed)\n")
        return rows
    except Exception as e:
        print(f"  Could not download index — {e}\n")
        return []


# ── Step 3: best-effort Box.com downloads for the rest ───────────────────────

def try_box_download(box_url: str, out_path: Path) -> bool:
    """Attempt to pull a direct file from a Box shared-link page.

    Box sometimes serves the raw file if '/download' is appended to a
    public shared link; other times it returns an HTML viewer page
    instead. We check the response looks like a PDF before saving.
    """
    try:
        dl_url = box_url.rstrip("/") + "/download"
        resp = requests.get(dl_url, headers=HEADERS, timeout=BOX_TIMEOUT, allow_redirects=True)
        content_type = resp.headers.get("Content-Type", "")
        if resp.status_code == 200 and (
            "pdf" in content_type.lower() or resp.content[:4] == b"%PDF"
        ):
            out_path.write_bytes(resp.content)
            return True
    except Exception:
        pass
    return False


def download_box_reports(rows: list[dict], pdf_dir: Path, max_attempts: int) -> int:
    if not rows:
        return 0

    print(f"  Step 3 — Best-effort Box.com downloads (attempting up to {max_attempts})")
    print("  This step commonly has a low success rate — that's expected, not a bug.\n")

    candidates = [r for r in rows if r.get("Link", "").startswith("https://app.box.com")]
    candidates = candidates[:max_attempts]

    ok = 0
    for row in tqdm(candidates, desc="  Box (best-effort)", ncols=80, unit="pdf"):
        filename = (row.get("Filename") or row.get("Title") or "report").strip()
        safe_name = "".join(c for c in filename if c not in '<>:"/\\|?*').strip()[:120]
        out_path = pdf_dir / f"{safe_name}.pdf"

        if out_path.exists():
            ok += 1
            continue

        if try_box_download(row["Link"], out_path):
            ok += 1
        time.sleep(REQUEST_DELAY)

    print(f"\n  Box reports saved        : {ok} / {len(candidates)} attempted "
          f"(low yield is expected — see docstring at top of this file)\n")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Digital Detective — Data Acquisition")
    print(f"  {'─'*45}")
    print(f"  Output: {PDF_DIR}\n")

    n_historical = download_historical(PDF_DIR)
    rows         = download_index(INDEX_CSV)
    n_box        = download_box_reports(rows, PDF_DIR, MAX_BOX_ATTEMPTS)

    total = len(list(PDF_DIR.glob("*.pdf")))

    print(f"  {'='*50}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"  {'='*50}")
    print(f"  Reliable (git-hosted)   : {n_historical}")
    print(f"  Best-effort (Box.com)   : {n_box}")
    print(f"  Total PDFs in folder    : {total}")
    print(f"  Report index saved to   : {INDEX_CSV}")
    if total < 20:
        print(f"\n  Tip: for a larger corpus, open links from {INDEX_CSV.name}")
        print(f"  in a browser and save additional PDFs into {PDF_DIR}/")
    print(f"\n  Next: run phase1_extract_text.py")
    print(f"  {'='*50}\n")


if __name__ == "__main__":
    main()
