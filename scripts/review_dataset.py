"""Human-operated review helper. It never silently approves source or gold records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl, utc_now, write_jsonl


def review_sources(rows, reviewer):
    for row in rows:
        print(f"\n{row['doc_id']} — {row['title']}\n{row['url']}\nHash: {row['content_hash']}")
        for paragraph in row["paragraphs"]:
            print(f"\n[{paragraph['paragraph_id']}] {paragraph['heading']}\n{paragraph['text']}")
        print("Check completeness, stable scope, extraction, source conflicts, and permitted use.")
        if input(f"Type approve {row['doc_id']} after inspection, or Enter to leave pending: ").strip() == f"approve {row['doc_id']}":
            row.update(review_status="human_reviewed", reviewer=reviewer, reviewed_at=utc_now())
    return rows


def bind_span(text, documents):
    # One required fact can have alternative supporting spans separated by |.
    alternatives = []
    for entry in text.split("|"):
        doc_id, paragraph_id, start, end = entry.strip().split(":")
        doc = documents[doc_id]
        paragraph = next(p for p in doc.paragraphs if p.paragraph_id == paragraph_id)
        start, end = int(start), int(end)
        if not 0 <= start < end <= len(paragraph.text):
            raise ValueError("Span is outside paragraph")
        print(f"\nSelected evidence: {paragraph.text[start:end]}")
        alternatives.append({"doc_id": doc_id, "paragraph_id": paragraph_id,
                             "start": start, "end": end, "content_hash": doc.content_hash})
    return alternatives


def review_questions(rows, reviewer, corpus):
    documents = {d.doc_id: d for d in load_documents(corpus)}
    for row in rows:
        print(f"\n{row['question_id']} [{row['split']}] {row['question']}\nLabel: {row['label']}")
        print("Draft answer:", row["reference_answer"])
        print("Required points:", row["required_points"])
        print("Edit an incorrect draft in the input JSONL before reviewing; Enter skips this case.")
        if input("Review this case now? Type review: ").strip() != "review":
            continue
        for doc_id in row.get("proposed_source_ids", []):
            if doc_id not in documents:
                print(f"Missing proposed source {doc_id}; do not approve without sufficient evidence.")
                continue
            for p in documents[doc_id].paragraphs:
                print(f"\n{doc_id}:{p.paragraph_id} ({len(p.text)} characters) [{p.heading}]\n{p.text}")
        groups = []
        if row["label"] == "answerable":
            print("Enter one evidence group per required fact. Format D01:p001:0:100.")
            print("Alternatives: D01:p001:0:100 | D02:p003:20:90. Empty input finishes groups.")
            while True:
                text = input("Evidence group: ").strip()
                if not text:
                    break
                try:
                    groups.append(bind_span(text, documents))
                except (ValueError, KeyError, StopIteration):
                    print("Invalid span. Use exact document/paragraph IDs and character offsets.")
            if not groups:
                print("No evidence selected; left pending.")
                continue
        else:
            print("Search/read the entire selected corpus; a retrieval miss alone is insufficient.")
            print("Reason:", row.get("unanswerable_reason"))
        if input(f"Type approve {row['question_id']} only after checking all gold fields: ").strip() == f"approve {row['question_id']}":
            row.update(evidence_groups=groups, review_status="human_reviewed",
                       reviewer=reviewer, reviewed_at=utc_now())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["sources", "questions"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Preserve it and choose a new reviewed version.")
    if args.kind == "questions" and not args.corpus:
        parser.error("Question review requires --corpus")
    rows = read_jsonl(args.input)
    try:
        if args.kind == "sources":
            # Validate hashes/shape without claiming human review.
            load_documents(args.input, allow_unreviewed=True)
            rows = review_sources(rows, args.reviewer)
        else:
            rows = review_questions(rows, args.reviewer, args.corpus)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted. Saving only review decisions already made.")
    finally:
        write_jsonl(args.output, rows)
    print(f"Saved {sum(r.get('review_status') == 'human_reviewed' for r in rows)}/{len(rows)} reviewed records.")


if __name__ == "__main__":
    main()
