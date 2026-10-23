"""Apply explicit, versioned source filters without rewriting historical snapshots."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import digest, file_hash, read_jsonl, utc_now, write_json, write_jsonl


def curate(source: Path, specification: Path, output: Path, manifest: Path, log: Path,
           snapshots: Path) -> dict:
    if any(p.exists() for p in [output, manifest, log]):
        raise ValueError("Choose new corpus, manifest and log versions; never overwrite")
    load_documents(source, allow_unreviewed=True)
    import json
    spec = json.loads(specification.read_text())
    originals = read_jsonl(source)
    rows = deepcopy(originals)
    by_id = {r["doc_id"]: r for r in rows}
    changes = []
    for operation in spec["operations"]:
        doc = by_id[operation["doc_id"]]
        if doc["content_hash"] != operation["expected_parent_hash"]:
            raise ValueError("Curation specification belongs to a different source snapshot")
        p = next(p for p in doc["paragraphs"] if p["paragraph_id"] == operation["paragraph_id"])
        before = p["text"]
        if operation["action"] == "drop":
            doc["paragraphs"].remove(p)
            after = ""
        elif operation["action"] == "keep_prefix_sentences":
            sentences = list(re.finditer(r".+?(?:[.!?](?=\s|$)|$)", before))
            after = before[:sentences[operation["count"] - 1].end()].strip()
            p["text"] = after
        else:
            raise ValueError("Unknown curation action")
        changes.append({**operation, "before_sha256": digest(before), "after_sha256": digest(after),
                        "before_characters": len(before), "after_characters": len(after)})
    for row in rows:
        parent_hash = row["content_hash"]
        row.update(version=spec["version"], parent_content_sha256=parent_hash,
                   source_url=row["url"], retrieved_at=row.get("fetched_at"),
                   content_hash=digest(row["paragraphs"]), curation_spec_sha256=file_hash(specification),
                   review_status="pending_human_review", reviewer=None, reviewed_at=None)
        row["content_sha256"] = row["content_hash"]
        raw_path = snapshots / f"{row['doc_id']}.txt"
        if file_hash(raw_path) != row["source_extract_hash"]:
            raise ValueError("Raw snapshot hash differs from source provenance")
        raw = raw_path.read_text()
        updated = re.search(r"^L(\d+): Updated (.+)$", raw, re.M)
        if updated and updated[2] == row.get("source_updated_at"):
            row["source_updated_at_evidence"] = {"snapshot_line": int(updated[1]),
                "snapshot_sha256": row["source_extract_hash"], "literal_value": updated[2]}
        else:
            row["source_updated_at"] = None
            row["source_updated_at_evidence"] = None
        live_ids = {p["paragraph_id"] for p in row["paragraphs"]}
        row["paragraph_provenance"] = [p for p in row["paragraph_provenance"]
                                        if p["paragraph_id"] in live_ids]
        row["context_groups"] = spec.get("context_groups", {}).get(row["doc_id"], [])
        row["known_conflicts"] = spec.get("known_conflicts", {}).get(row["doc_id"], [])
    public = [{k: v for k, v in r.items() if k != "paragraphs"} for r in rows]
    result = {"version": spec["version"], "created_at": utc_now(),
              "parent_file_sha256": file_hash(source), "spec_sha256": file_hash(specification),
              "documents": len(rows), "paragraphs_before": sum(len(r["paragraphs"]) for r in originals),
              "paragraphs_after": sum(len(r["paragraphs"]) for r in rows), "changes": changes,
              "review_status": "pending_human_review"}
    write_jsonl(output, rows)
    load_documents(output, allow_unreviewed=True)
    write_jsonl(manifest, public)
    write_json(log, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "spec", "output", "manifest", "log", "snapshots"]:
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    result = curate(a.input, a.spec, a.output, a.manifest, a.log, a.snapshots)
    print({k: result[k] for k in ["version", "documents", "paragraphs_before", "paragraphs_after"]})
