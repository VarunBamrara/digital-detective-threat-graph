"""
Digital Detective — Phase 2: LLM Triple Extraction (Final)
===========================================================
Sends each text chunk to Groq and extracts
Subject → Predicate → Object triples for the knowledge graph.

Setup (one-time):
    pip install groq tqdm
    1. Go to https://console.groq.com  →  sign up free
    2. Create an API key
    3. Paste it below in GROQ_API_KEY

Usage:
    python phase2_extract_triples.py

Output:
    digital_detective_data/triples/           ← per-PDF triple files
    digital_detective_data/all_triples.json   ← merged, deduplicated
    digital_detective_data/triples_summary.json
"""

import re
import json
import time
from pathlib import Path
from tqdm import tqdm
from groq import Groq
from project_config import get_env

# ── CONFIG ────────────────────────────────────────────────────────────────────
GROQ_API_KEY        = get_env("GROQ_API_KEY", "")
MAX_CHUNKS_PER_FILE = 20    # chunks per PDF — set None for full run later

# ── Fixed settings (do not change) ───────────────────────────────────────────
MODEL        = "llama-3.1-8b-instant"   # current Groq free model, fast + reliable
DELAY_BASE   = 5.0                       # seconds between API calls
DEBUG_CHUNKS = 0                         # set to 2 to see raw LLM output for first 2 chunks

CHUNKS_DIR   = Path("digital_detective_data/extracted_chunks")
TRIPLES_DIR  = Path("digital_detective_data/triples")
ALL_TRIPLES  = Path("digital_detective_data/all_triples.json")
SUMMARY_FILE = Path("digital_detective_data/triples_summary.json")

# ── Valid schema ──────────────────────────────────────────────────────────────
VALID_ENTITY_TYPES = {
    "ThreatActor", "Malware", "Tool", "CVE", "IPAddress",
    "Domain", "Industry", "Country", "TTP", "Campaign",
    "Organization", "FileHash", "Vulnerability"
}
VALID_PREDICATES = {
    "USES", "TARGETS", "EXPLOITS", "ATTRIBUTED_TO", "PART_OF",
    "COMMUNICATES_WITH", "DOWNLOADS", "DROPS", "RELATED_TO",
    "OPERATES_IN", "COMPROMISES", "DELIVERS", "ASSOCIATED_WITH"
}

# ── Prompt with worked examples ───────────────────────────────────────────────
SYSTEM_PROMPT = """You are a cybersecurity knowledge graph builder. Extract relationship triples from threat intelligence text.

ALWAYS output a JSON array. NEVER output empty array [] if the text mentions any threat actors, malware, tools, CVEs, IPs, countries, or industries.

ENTITY TYPES (use exactly these strings):
ThreatActor, Malware, Tool, CVE, IPAddress, Domain, Industry, Country, TTP, Campaign, Organization, FileHash, Vulnerability

PREDICATES (use exactly these strings):
USES, TARGETS, EXPLOITS, ATTRIBUTED_TO, PART_OF, COMMUNICATES_WITH, DOWNLOADS, DROPS, RELATED_TO, OPERATES_IN, COMPROMISES, DELIVERS, ASSOCIATED_WITH

OUTPUT FORMAT — return ONLY a raw JSON array, no markdown, no explanation:
[{"subject":"Name","subject_type":"Type","predicate":"PREDICATE","object":"Name","object_type":"Type","confidence":0.9}]

EXAMPLES:
Text: "APT1, also known as Comment Crew, is a Chinese threat group that uses the malware WEBC2 to target aerospace companies."
Output: [{"subject":"APT1","subject_type":"ThreatActor","predicate":"USES","object":"WEBC2","object_type":"Malware","confidence":0.95},{"subject":"APT1","subject_type":"ThreatActor","predicate":"TARGETS","object":"Aerospace","object_type":"Industry","confidence":0.95},{"subject":"APT1","subject_type":"ThreatActor","predicate":"ATTRIBUTED_TO","object":"China","object_type":"Country","confidence":0.85}]

Text: "The Lazarus Group delivered the BLINDINGCAN backdoor via spear-phishing emails to defense contractors in Europe."
Output: [{"subject":"Lazarus Group","subject_type":"ThreatActor","predicate":"DELIVERS","object":"BLINDINGCAN","object_type":"Malware","confidence":0.95},{"subject":"Lazarus Group","subject_type":"ThreatActor","predicate":"TARGETS","object":"Defense","object_type":"Industry","confidence":0.9},{"subject":"Lazarus Group","subject_type":"ThreatActor","predicate":"OPERATES_IN","object":"Europe","object_type":"Country","confidence":0.8}]

Text: "Stuxnet exploited CVE-2010-2568 and communicated with command-and-control servers at 188.120.229.232."
Output: [{"subject":"Stuxnet","subject_type":"Malware","predicate":"EXPLOITS","object":"CVE-2010-2568","object_type":"CVE","confidence":0.95},{"subject":"Stuxnet","subject_type":"Malware","predicate":"COMMUNICATES_WITH","object":"188.120.229.232","object_type":"IPAddress","confidence":0.9}]

Now extract triples from the text below. Return ONLY the JSON array:"""


# ── LLM call ──────────────────────────────────────────────────────────────────
_debug_count = 0

def extract_triples_from_chunk(client: Groq, text: str) -> list[dict]:
    global _debug_count

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text}
        ],
        temperature=0.0,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()

    # Debug output for first N chunks
    if _debug_count < DEBUG_CHUNKS:
        _debug_count += 1
        print(f"\n  [DEBUG {_debug_count}] LLM output:\n  {raw[:500]}\n")

    # Strip markdown fences if model adds them
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$",          "", raw, flags=re.MULTILINE).strip()

    # Extract JSON array from anywhere in the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        triples = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    if not isinstance(triples, list):
        return []

    # Validate and normalise each triple
    valid = []
    for t in triples:
        if not all(k in t for k in ["subject", "subject_type", "predicate", "object", "object_type"]):
            continue
        pred  = t["predicate"].upper().replace(" ", "_")
        stype = t["subject_type"].strip()
        otype = t["object_type"].strip()
        subj  = t["subject"].strip()
        obj   = t["object"].strip()

        if pred  not in VALID_PREDICATES:    continue
        if stype not in VALID_ENTITY_TYPES:  continue
        if otype not in VALID_ENTITY_TYPES:  continue
        if len(subj) < 2 or len(obj) < 2:   continue

        valid.append({
            "subject":      subj,
            "subject_type": stype,
            "predicate":    pred,
            "object":       obj,
            "object_type":  otype,
            "confidence":   float(t.get("confidence", 0.8))
        })

    return valid


# ── Process one PDF ───────────────────────────────────────────────────────────
def process_file(client: Groq, chunk_file: Path) -> dict:
    data   = json.loads(chunk_file.read_text(encoding="utf-8"))
    chunks = data.get("chunks", [])
    if MAX_CHUNKS_PER_FILE:
        chunks = chunks[:MAX_CHUNKS_PER_FILE]

    all_triples  = []
    errors       = 0
    rate_hits    = 0

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if len(text) < 80:
            continue
        try:
            triples = extract_triples_from_chunk(client, text)
            for t in triples:
                t["source_file"] = chunk_file.stem
                t["source_page"] = chunk.get("page", 0)
            all_triples.extend(triples)
            rate_hits = 0   # reset backoff counter on success

        except Exception as e:
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in msg or "too many" in msg:
                rate_hits += 1
                wait = min(30 * (2 ** (rate_hits - 1)), 300)  # 30 → 60 → 120 → 300s max
                tqdm.write(f"  Rate limit (hit #{rate_hits}) — waiting {wait}s ...")
                time.sleep(wait)
                # Retry the same chunk once after waiting
                try:
                    triples = extract_triples_from_chunk(client, text)
                    for t in triples:
                        t["source_file"] = chunk_file.stem
                        t["source_page"] = chunk.get("page", 0)
                    all_triples.extend(triples)
                except Exception:
                    errors += 1
            elif "decommissioned" in msg or "model" in msg:
                tqdm.write(f"  Model error: {str(e)[:120]}")
                tqdm.write("  Fix: update MODEL variable in the script")
                raise   # stop immediately — wrong model
            else:
                errors += 1
                tqdm.write(f"  Error on chunk: {str(e)[:100]}")

        time.sleep(DELAY_BASE)

    return {
        "filename":         chunk_file.name,
        "chunks_processed": len(chunks),
        "triples_found":    len(all_triples),
        "errors":           errors,
        "triples":          all_triples
    }


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate(triples: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for t in triples:
        key = (t["subject"].lower(), t["predicate"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not GROQ_API_KEY:
        print("\n  ERROR: GROQ_API_KEY is not set.")
        print("  Create .env from .env.example and set GROQ_API_KEY.")
        print("  Get a free key at: https://console.groq.com\n")
        return

    TRIPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old zero-triple cached files so they get re-processed
    cleared = 0
    for f in TRIPLES_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("triples_found", 0) == 0:
                f.unlink()
                cleared += 1
        except Exception:
            pass
    if cleared:
        print(f"  Cleared {cleared} empty cached files — will re-process\n")

    client      = Groq(api_key=GROQ_API_KEY)
    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))

    if not chunk_files:
        print(f"  No chunk files found in {CHUNKS_DIR}")
        print("  Run phase1_extract_text.py first.")
        return

    mode = f"first {MAX_CHUNKS_PER_FILE} chunks" if MAX_CHUNKS_PER_FILE else "ALL chunks"
    print(f"\n  Phase 2 — LLM Triple Extraction")
    print(f"  {'─'*45}")
    print(f"  Model     : {MODEL}")
    print(f"  Files     : {len(chunk_files)} PDFs")
    print(f"  Mode      : {mode} per PDF")
    print(f"  Output    : {TRIPLES_DIR}/")
    print()

    summary = {
        "model": MODEL,
        "total_files": len(chunk_files),
        "processed": 0, "skipped": 0,
        "total_triples_raw": 0,
        "total_triples_deduped": 0,
        "files": []
    }
    all_triples = []

    for chunk_file in tqdm(chunk_files, desc="  Extracting", ncols=80, unit="pdf"):
        out_path = TRIPLES_DIR / chunk_file.name

        # Resume: skip already-processed files
        if out_path.exists():
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            ex_t = existing.get("triples", [])
            all_triples.extend(ex_t)
            summary["skipped"] += 1
            summary["total_triples_raw"] += len(ex_t)
            summary["files"].append({
                "file": chunk_file.name,
                "status": "cached",
                "triples": len(ex_t)
            })
            continue

        result = process_file(client, chunk_file)

        # Save per-PDF result
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        all_triples.extend(result["triples"])
        summary["processed"]         += 1
        summary["total_triples_raw"] += result["triples_found"]
        summary["files"].append({
            "file":    chunk_file.name,
            "status":  "ok",
            "chunks":  result["chunks_processed"],
            "triples": result["triples_found"],
            "errors":  result["errors"]
        })

        icon = "✓" if result["triples_found"] > 0 else "○"
        tqdm.write(f"  {icon}  {chunk_file.stem[:48]:<48} → {result['triples_found']} triples")

    # Merge and deduplicate everything
    unique = deduplicate(all_triples)
    summary["total_triples_deduped"] = len(unique)

    ALL_TRIPLES.write_text(
        json.dumps(unique, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # Top 5 most productive files
    top5 = sorted(
        [f for f in summary["files"] if f.get("triples", 0) > 0],
        key=lambda x: x["triples"], reverse=True
    )[:5]

    print(f"\n  {'='*52}")
    print(f"  PHASE 2 COMPLETE")
    print(f"  {'='*52}")
    print(f"  PDFs processed      : {summary['processed']}  (+{summary['skipped']} cached/skipped)")
    print(f"  Raw triples         : {summary['total_triples_raw']}")
    print(f"  After deduplication : {summary['total_triples_deduped']}")
    if top5:
        print(f"\n  Top 5 most productive reports:")
        for f in top5:
            print(f"    {f['triples']:>4} triples — {f['file']}")
    print(f"\n  All triples saved → {ALL_TRIPLES}")
    print(f"  Summary saved     → {SUMMARY_FILE}")
    print(f"\n  Next step: run phase3_build_graph.py  (Neo4j knowledge graph)")
    print(f"  {'='*52}\n")


if __name__ == "__main__":
    main()