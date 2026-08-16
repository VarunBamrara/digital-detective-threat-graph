"""
Digital Detective — Phase 1: PDF Text Extraction & Cleaning
============================================================
Reads all APTNotes PDFs using PyMuPDF, cleans the text,
and outputs paragraph-level chunks as JSON — ready for
the LLM triple extraction pipeline in Phase 2.

Usage:
    pip install pymupdf tqdm
    python phase1_extract_text.py

Output:
    digital_detective_data/extracted_chunks/
        ├── Mandiant_APT1_Report.json
        ├── icefog.json
        ├── ...
    digital_detective_data/extraction_summary.json
"""

import re
import json
import unicodedata
from pathlib import Path
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Run:  pip install pymupdf")
    raise

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR        = Path("digital_detective_data/aptnotes_pdfs")
OUTPUT_DIR     = Path("digital_detective_data/extracted_chunks")
SUMMARY_FILE   = Path("digital_detective_data/extraction_summary.json")

MIN_CHUNK_LEN  = 80    # ignore chunks shorter than this (headers, page nums, etc.)
MAX_CHUNK_LEN  = 1200  # split chunks longer than this at sentence boundaries
MIN_PAGE_TEXT  = 30    # skip pages with fewer chars (blank/image-only pages)

# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(raw: str) -> str:
    """Normalize and clean raw PDF text."""
    # Normalize unicode (handles ligatures like ﬁ → fi, curly quotes, etc.)
    text = unicodedata.normalize("NFKC", raw)

    # Remove null bytes and other control characters (keep \n \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Collapse repeated whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)

    # Merge hyphenated line-breaks: "mal-\nware" → "malware"
    text = re.sub(r"-\n(\w)", r"\1", text)

    # Collapse more than 2 consecutive newlines into exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just page numbers (e.g. "- 12 -", "Page 5 of 30")
    text = re.sub(r"(?m)^[\s\-]*(?:page\s*)?\d+(?:\s*of\s*\d+)?[\s\-]*$", "", text, flags=re.IGNORECASE)

    # Remove lines that are only punctuation / special chars (table borders, etc.)
    text = re.sub(r"(?m)^[^a-zA-Z0-9]{0,3}$", "", text)

    return text.strip()


def is_noise_chunk(text: str) -> bool:
    """Return True for chunks that carry no useful intelligence text."""
    t = text.strip()
    if len(t) < MIN_CHUNK_LEN:
        return True
    # Mostly numbers / hex (IOC lists, hashes — useful later but not for NLP)
    non_alpha = sum(1 for c in t if not c.isalpha() and not c.isspace())
    if len(t) > 0 and non_alpha / len(t) > 0.65:
        return True
    # Table of contents patterns
    if re.search(r"\.{5,}", t):
        return True
    return False


def split_into_chunks(text: str) -> list[str]:
    """Split a page/block into paragraph-level chunks, respecting MAX_CHUNK_LEN."""
    # First split on paragraph breaks
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= MAX_CHUNK_LEN:
            chunks.append(para)
        else:
            # Split long paragraphs at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 <= MAX_CHUNK_LEN:
                    current = (current + " " + sent).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            if current:
                chunks.append(current)
    return chunks

# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_pdf(pdf_path: Path) -> dict:
    """Extract and clean all text from a single PDF. Returns a result dict."""
    result = {
        "filename":    pdf_path.name,
        "pages":       0,
        "chunks":      [],
        "skipped_pages": 0,
        "errors":      []
    }

    try:
        doc = fitz.open(pdf_path)
        result["pages"] = len(doc)
        all_chunks = []

        for page_num, page in enumerate(doc, start=1):
            try:
                raw = page.get_text("text")
            except Exception as e:
                result["errors"].append(f"Page {page_num}: {e}")
                continue

            if len(raw.strip()) < MIN_PAGE_TEXT:
                result["skipped_pages"] += 1
                continue

            cleaned = clean_text(raw)
            page_chunks = split_into_chunks(cleaned)

            for chunk in page_chunks:
                if not is_noise_chunk(chunk):
                    all_chunks.append({
                        "page":  page_num,
                        "text":  chunk,
                        "chars": len(chunk)
                    })

        doc.close()
        result["chunks"] = all_chunks

    except Exception as e:
        result["errors"].append(f"Could not open PDF: {e}")

    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Run download.py first.")
        return

    print(f"\n  Phase 1 — PDF Text Extraction")
    print(f"  {'─'*40}")
    print(f"  Input  : {PDF_DIR}  ({len(pdfs)} PDFs)")
    print(f"  Output : {OUTPUT_DIR}")
    print()

    summary = {
        "total_pdfs":        len(pdfs),
        "successful":        0,
        "failed":            0,
        "total_chunks":      0,
        "total_pages":       0,
        "skipped_pages":     0,
        "files":             []
    }

    for pdf_path in tqdm(pdfs, desc="  Extracting", ncols=80, unit="pdf"):
        out_path = OUTPUT_DIR / (pdf_path.stem + ".json")

        # Skip already-processed files
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            summary["successful"]    += 1
            summary["total_chunks"]  += len(existing.get("chunks", []))
            summary["total_pages"]   += existing.get("pages", 0)
            summary["skipped_pages"] += existing.get("skipped_pages", 0)
            summary["files"].append({
                "file":   pdf_path.name,
                "status": "skipped (already done)",
                "chunks": len(existing.get("chunks", []))
            })
            continue

        result = extract_pdf(pdf_path)
        n_chunks = len(result["chunks"])

        if result["errors"] and n_chunks == 0:
            summary["failed"] += 1
            status = "failed"
            tqdm.write(f"  ✗  {pdf_path.name} — {result['errors'][0]}")
        else:
            summary["successful"]    += 1
            summary["total_chunks"]  += n_chunks
            summary["total_pages"]   += result["pages"]
            summary["skipped_pages"] += result["skipped_pages"]
            status = "ok"

            # Save extracted chunks
            out_path.write_text(
                json.dumps(result, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )

        summary["files"].append({
            "file":   pdf_path.name,
            "status": status,
            "pages":  result["pages"],
            "chunks": n_chunks,
            "errors": result["errors"]
        })

    # Save summary
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n  {'='*50}")
    print(f"  PHASE 1 COMPLETE")
    print(f"  {'='*50}")
    print(f"  PDFs processed  : {summary['successful']} / {summary['total_pdfs']}")
    print(f"  Failed          : {summary['failed']}")
    print(f"  Total pages     : {summary['total_pages']}")
    print(f"  Skipped pages   : {summary['skipped_pages']}  (blank/image-only)")
    print(f"  Total chunks    : {summary['total_chunks']}  ← these feed Phase 2")
    print()

    # Show top 5 richest files (most chunks = most text)
    top5 = sorted(
        [f for f in summary["files"] if f.get("chunks", 0) > 0],
        key=lambda x: x["chunks"], reverse=True
    )[:5]
    if top5:
        print(f"  Top 5 richest reports (most text chunks):")
        for f in top5:
            print(f"    {f['chunks']:>4} chunks — {f['file']}")

    avg = summary["total_chunks"] // max(summary["successful"], 1)
    print(f"\n  Avg chunks/PDF  : {avg}")
    print(f"  Summary saved   : {SUMMARY_FILE}")
    print(f"\n  Next: run phase2_extract_triples.py")
    print(f"  {'='*50}\n")


if __name__ == "__main__":
    main()
