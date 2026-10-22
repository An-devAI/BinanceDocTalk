"""CLI entry points. Paid inference is always an explicit command."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from .answering import AnswerService
from .evaluation import (freeze_experiment, make_grade_sheet, retrieval_evaluation,
                         run_experiment, score_predictions, validate_questions)
from .ingest import collect, load_documents
from .io import load_env, read_jsonl, write_json, write_jsonl
from .provider import OpenAIProvider
from .retrieval import Index, build_index
from .schemas import Config

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT = (_SOURCE_ROOT if (_SOURCE_ROOT / "data/demo/notes.jsonl").is_file()
        else Path.cwd())


def demo_service() -> AnswerService:
    config = replace(Config(), tokenizer="regex_word_v1", top_k=3)
    docs = load_documents(ROOT / "data/demo/notes.jsonl")
    return AnswerService(build_index(docs, config), config)


def benchmark_status() -> dict:
    rows = []
    for split in ["smoke", "dev", "test"]:
        reviewed = ROOT / f"data/eval/v1/{split}.jsonl"
        path = reviewed if reviewed.exists() else ROOT / f"data/eval/drafts/{split}.jsonl"
        rows.extend(read_jsonl(path))
    return {"questions": len(rows),
            "human_reviewed": sum(r.get("review_status") == "human_reviewed" for r in rows),
            "note": "Owner-declared review status, not independent certification"}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="BinanceDocTalk — independent educational RAG")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check environment without making an API request")
    demo = sub.add_parser("demo", help="Original-notes retrieval preview; no API, no official corpus")
    demo.add_argument("--question", default="Why does a stop-limit order not guarantee execution?")
    demo.add_argument("--web", action="store_true")
    demo.add_argument("--port", type=int, default=8765)
    collect_cmd = sub.add_parser("collect")
    collect_cmd.add_argument("--manifest", type=Path, default=Path("data/manifest.v1.jsonl"))
    collect_cmd.add_argument("--output", type=Path, default=Path("data/local/corpus.v1.jsonl"))
    collect_cmd.add_argument("--doc-id", action="append")
    collect_cmd.add_argument("--acknowledge-source-terms", action="store_true")
    build = sub.add_parser("build")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--output", type=Path, default=Path("data/local/index.v1.json"))
    build.add_argument("--config", type=Path, default=Path("configs/rag.toml"))
    build.add_argument("--lexical-only", action="store_true")
    build.add_argument("--allow-unreviewed", action="store_true")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--config", type=Path, default=Path("configs/rag.toml"))
    ask.add_argument("--index", type=Path, default=Path("data/local/index.v1.json"))
    ask.add_argument("--ledger", type=Path, default=Path("artifacts/local/budget.json"))
    audit = sub.add_parser("audit-benchmark")
    audit.add_argument("--directory", type=Path, default=Path("data/eval/drafts"))
    audit.add_argument("--require-reviewed", action="store_true")
    audit.add_argument("--corpus", type=Path)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--corpus", type=Path, required=True)
    freeze.add_argument("--index", type=Path, required=True)
    freeze.add_argument("--questions", type=Path, default=Path("data/eval/drafts"))
    freeze.add_argument("--output", type=Path, default=Path("artifacts/local/freeze.v1.json"))
    freeze.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.toml"))
    freeze.add_argument("--rag-config", type=Path, default=Path("configs/rag.toml"))
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--questions", type=Path, required=True)
    evaluate.add_argument("--index", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--split", choices=["smoke", "dev", "test"], required=True)
    evaluate.add_argument("--freeze", type=Path)
    evaluate.add_argument("--allow-draft", action="store_true")
    evaluate.add_argument("--baseline-config", type=Path, default=Path("configs/baseline.toml"))
    evaluate.add_argument("--rag-config", type=Path, default=Path("configs/rag.toml"))
    evaluate.add_argument("--ledger", type=Path, default=Path("artifacts/local/budget.json"))
    grade = sub.add_parser("grade-sheet")
    grade.add_argument("--predictions", type=Path, required=True)
    grade.add_argument("--questions", type=Path, required=True)
    grade.add_argument("--output", type=Path, required=True)
    report = sub.add_parser("report")
    report.add_argument("--predictions", type=Path, required=True)
    report.add_argument("--questions", type=Path, required=True)
    report.add_argument("--grades", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    retrieval = sub.add_parser("retrieval-eval")
    retrieval.add_argument("--questions", type=Path, required=True)
    retrieval.add_argument("--index", type=Path, required=True)
    retrieval.add_argument("--config", type=Path, default=Path("configs/rag.toml"))
    retrieval.add_argument("--dense", action="store_true")
    retrieval.add_argument("--output", type=Path, required=True)
    web = sub.add_parser("serve", help="Local UI with optional live mode")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--index", type=Path, default=Path("data/local/index.v1.json"))
    web.add_argument("--config", type=Path, default=Path("configs/rag.toml"))
    web.add_argument("--report", type=Path, help="Optional human-graded frozen metrics JSON")
    evidence = sub.add_parser("evidence", help="Inspect a local paragraph and print a span reference")
    evidence.add_argument("--corpus", type=Path, required=True)
    evidence.add_argument("--doc-id", required=True)
    evidence.add_argument("--paragraph-id", required=True)
    evidence.add_argument("--start", type=int, default=0)
    evidence.add_argument("--end", type=int)
    return result


def execute(args) -> object:
    load_env()
    if args.command == "doctor":
        return {"python": sys.version.split()[0], "api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                "network_tested": False, "live_model_tested": False,
                "official_corpus_present": Path("data/local/corpus.v1.jsonl").exists(),
                "note": "Doctor makes no paid requests and does not print secrets."}
    if args.command == "demo":
        if args.web:
            from .web import serve
            serve(args.port)
            return None
        return demo_service().ask(args.question, demo=True)
    if args.command == "collect":
        return collect(args.manifest, args.output, args.acknowledge_source_terms, args.doc_id)
    if args.command == "build":
        if args.output.exists():
            raise ValueError("Index already exists; choose a new version")
        documents = load_documents(args.corpus, args.allow_unreviewed)
        config = Config.load(args.config)
        provider = None if args.lexical_only else OpenAIProvider(config)
        index = build_index(documents, config, provider,
                            reviewed=all(r.get("review_status") == "human_reviewed"
                                         for r in read_jsonl(args.corpus)))
        index.save(args.output)
        return index.metadata
    if args.command == "ask":
        config = Config.load(args.config)
        provider = OpenAIProvider(config, args.ledger)
        index = Index.load(args.index) if config.system == "rag" else None
        return AnswerService(index, config, provider).ask(args.question)
    if args.command == "audit-benchmark":
        rows = [r for split in ["smoke", "dev", "test"]
                for r in read_jsonl(args.directory / f"{split}.jsonl")]
        documents = load_documents(args.corpus) if args.corpus else None
        if args.require_reviewed and documents is None:
            raise ValueError("Reviewed audit needs --corpus to validate evidence")
        return validate_questions(rows, args.require_reviewed, documents)
    if args.command == "freeze":
        return freeze_experiment(ROOT, args.corpus, args.index,
            [args.questions / f"{s}.jsonl" for s in ["smoke", "dev", "test"]],
            [args.baseline_config, args.rag_config], args.output)
    if args.command == "evaluate":
        return run_experiment(ROOT, read_jsonl(args.questions), Index.load(args.index),
            [Config.load(args.baseline_config), Config.load(args.rag_config)],
            lambda config: OpenAIProvider(config, args.ledger), args.output, args.split,
            args.freeze, args.allow_draft)
    if args.command == "grade-sheet":
        if args.output.exists():
            raise ValueError("Grade sheet already exists; do not overwrite human annotations")
        sheet = make_grade_sheet(read_jsonl(args.predictions), read_jsonl(args.questions))
        write_jsonl(args.output, sheet)
        return {"outputs_to_review": len(sheet), "output": str(args.output)}
    if args.command == "report":
        report = score_predictions(read_jsonl(args.predictions), read_jsonl(args.grades),
                                   read_jsonl(args.questions))
        write_json(args.output, report)
        return report
    if args.command == "retrieval-eval":
        provider = OpenAIProvider(Config.load(args.config)) if args.dense else None
        report = retrieval_evaluation(Index.load(args.index), read_jsonl(args.questions), provider)
        write_json(args.output, report)
        return report
    if args.command == "serve":
        from .web import serve
        serve(args.port, args.index, args.config, args.report)
        return None
    if args.command == "evidence":
        documents = load_documents(args.corpus, allow_unreviewed=True)
        doc = next((d for d in documents if d.doc_id == args.doc_id), None)
        if doc is None:
            raise ValueError("Unknown document")
        paragraph = next((p for p in doc.paragraphs if p.paragraph_id == args.paragraph_id), None)
        if paragraph is None:
            raise ValueError("Unknown paragraph")
        end = args.end if args.end is not None else len(paragraph.text)
        if not 0 <= args.start < end <= len(paragraph.text):
            raise ValueError("Invalid character range")
        return {"text": paragraph.text[args.start:end], "heading": paragraph.heading,
                "reference": {"doc_id": doc.doc_id, "paragraph_id": paragraph.paragraph_id,
                              "start": args.start, "end": end, "content_hash": doc.content_hash},
                "note": "Inspect this evidence before marking the question human_reviewed."}
    raise ValueError("Unknown command")


def main(argv: list[str] | None = None) -> int:
    try:
        output = execute(parser().parse_args(argv))
        if output is not None:
            print(json.dumps(output, indent=2, ensure_ascii=False))
        return 1 if isinstance(output, dict) and (output.get("status") == "error" or output.get("failures")) else 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}),
              file=sys.stderr)
        return 1
