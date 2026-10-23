"""Human-operated review helper. It never silently approves source or gold records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl, utc_now, write_jsonl
from binance_doctalk.benchmark import corpus_fingerprint, scope_policy, validate_v2
from binance_doctalk.evaluation import validate_questions


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
        print("Test type:", row.get("test_type"), "Family:", row["family_id"])
        if row.get("policy_basis"):
            print("Application policy (not source evidence):", scope_policy())
            print("Policy basis:", row["policy_basis"])
        if row.get("corpus_audit"):
            print("Whole-corpus audit:", row["corpus_audit"])
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
        elif row["label"] == "insufficient_information":
            print("Search/read the entire selected corpus; a retrieval miss alone is insufficient.")
            print("Reason:", row.get("unanswerable_reason"))
        else:
            print("Check the frozen application policy. Out-of-scope is not a corpus absence claim.")
        if input(f"Type approve {row['question_id']} only after checking all gold fields: ").strip() == f"approve {row['question_id']}":
            candidate = dict(row, evidence_groups=groups, review_status="human_reviewed",
                             reviewer=reviewer, reviewed_at=utc_now())
            try:
                validate_questions([candidate], True, list(documents.values()))
            except ValueError as exc:
                print("Not approved:", exc)
                continue
            row.update(candidate)
    return rows


def review_audits(rows, reviewer, corpus):
    documents = load_documents(corpus)
    expected = {d.doc_id: d.content_hash for d in documents}
    active = []
    print("Frozen scope policy:", scope_policy())
    for row in rows:
        if row["label"] != "insufficient_information":
            continue
        audit = row.get("corpus_audit", {})
        entries = audit.get("documents", [])
        if (audit.get("corpus_hash") != corpus_fingerprint(documents)
                or len(entries) != len(expected)
                or {e.get("doc_id"): e.get("content_hash") for e in entries} != expected):
            print(row["question_id"], "stale/incomplete audit; regenerate against the selected corpus")
            continue
        print(row["question_id"], row["question"])
        print("Proposed missing information:", audit.get("missing_information"))
        if input("Confirm this conceptual question is in scope. Type in_scope, or Enter to skip: ").strip() == "in_scope":
            active.append(row)
    # Read each source once per split, then make a separate decision for each active question.
    for doc in documents if active else []:
        print(f"\n{doc.doc_id} — {doc.title} — {doc.url}")
        for p in doc.paragraphs:
            print(f"[{p.paragraph_id}] {p.heading}\n{p.text}")
        for row in active:
            audit = row["corpus_audit"]
            entry = next(e for e in audit["documents"] if e["doc_id"] == doc.doc_id)
            print(row["question_id"], row["question"])
            print("Lexical candidates only:", entry.get("matched_paragraph_ids"))
            print("Existing decision:", entry.get("assessment"), entry.get("notes"))
            decision = input("After reading ALL paragraphs: absent / partial / sufficient; Enter keeps pending or prior decision: ").strip()
            if decision not in {"absent", "partial", "sufficient"}:
                continue
            notes = input("Explain what this source establishes and what is still missing: ").strip()
            if notes:
                entry.update(assessment=decision, notes=notes, reviewer=reviewer, reviewed_at=utc_now())
                audit.update(review_status="pending_human_review", reviewer=None, reviewed_at=None)
                row.update(review_status="pending_human_review", reviewer=None, reviewed_at=None)
    for row in active:
        audit = row["corpus_audit"]
        if any(e.get("assessment") not in {"absent", "partial"} or not e.get("notes")
               for e in audit["documents"]):
            print(row["question_id"], "Incomplete audit, or sufficient evidence found. Correct/relabel; no approval.")
            continue
        if input(f"Type approve audit {row['question_id']} after whole-corpus inspection: ").strip() == f"approve audit {row['question_id']}":
            candidate = dict(audit, review_status="human_reviewed", reviewer=reviewer, reviewed_at=utc_now())
            validate_v2(dict(row, corpus_audit=candidate), documents, True)
            audit.update(candidate)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["sources", "questions", "audits"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists. Preserve it and choose a new reviewed version.")
    if args.kind != "sources" and not args.corpus:
        parser.error("Question review requires --corpus")
    rows = read_jsonl(args.input)
    try:
        if args.kind == "sources":
            # Validate hashes/shape without claiming human review.
            load_documents(args.input, allow_unreviewed=True)
            rows = review_sources(rows, args.reviewer)
        elif args.kind == "audits":
            rows = review_audits(rows, args.reviewer, args.corpus)
        else:
            rows = review_questions(rows, args.reviewer, args.corpus)
    except (EOFError, KeyboardInterrupt):
        print("\nInterrupted. Saving only review decisions already made.")
    finally:
        write_jsonl(args.output, rows)
    print(f"Saved {sum(r.get('review_status') == 'human_reviewed' for r in rows)}/{len(rows)} reviewed records.")


if __name__ == "__main__":
    main()
