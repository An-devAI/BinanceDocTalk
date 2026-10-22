"""Behavioral tests on original/synthetic data. These are NOT quality benchmarks."""

from dataclasses import asdict, replace
from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from binance_doctalk.answering import AnswerService, BASELINE_PROMPT, RAG_PROMPT
from binance_doctalk.chunking import chunk_document
from binance_doctalk.cli import ROOT, demo_service, main
from binance_doctalk.evaluation import (evidence_covered, make_grade_sheet, score_predictions,
                                       validate_questions, run_experiment)
from binance_doctalk.ingest import allowed_url, extract_html, load_documents
from binance_doctalk.io import digest, read_jsonl, write_jsonl
from binance_doctalk.provider import BudgetError, Ledger, OpenAIProvider, ProviderError
from binance_doctalk.retrieval import Index, build_index, normalize
from binance_doctalk.schemas import Config, Document, Paragraph, validate_answer


def fixture_doc(kind="synthetic_demo"):
    paragraphs = [Paragraph("p001", "First", "alpha beta gamma delta epsilon zeta"),
                  Paragraph("p002", "First", "eta theta iota kappa lambda mu"),
                  Paragraph("p003", "Other", "nectarine orange pear kiwi")]
    return Document("TEST", "Synthetic unit-test document",
                    "https://www.binance.com/en/academy/articles/unit-test" if kind == "official" else "demo:test",
                    "2026-01-01T00:00:00+00:00" if kind == "official" else None,
                    digest([asdict(p) for p in paragraphs]), tuple(paragraphs), kind)


class IngestionTests(unittest.TestCase):
    def test_allowlist_rejects_other_hosts_and_private_paths(self):
        self.assertTrue(allowed_url("https://www.binance.com/en/academy/articles/a"))
        for url in ["http://www.binance.com/en/academy/articles/a", "https://evil.test/en/academy/a",
                    "https://www.binance.com/en/my/account", "https://user@www.binance.com/en/academy/a",
                    "https://www.binance.com:8443/en/academy/a"]:
            self.assertFalse(allowed_url(url), url)

    def test_extraction_filters_boilerplate_and_retains_heading(self):
        html = ('<nav><p>irrelevant navigation text</p></nav><h1>Article</h1>'
                '<h2>Keep</h2><p>A sufficiently long paragraph with <b>inline</b> text.</p>'
                '<h2>Drop</h2><p>This paragraph must be excluded.</p>'
                '<h2>Related Articles</h2><p>No unrelated marketing text should survive.</p>')
        result = extract_html(html, {"doc_id": "T", "url": "https://www.binance.com/en/academy/a",
                                     "include_headings": ["Keep"]})
        self.assertEqual(len(result["paragraphs"]), 1)
        self.assertIn("inline text", result["paragraphs"][0]["text"])
        self.assertEqual(result["review_status"], "pending_human_review")

    def test_block_page_fails_instead_of_empty_corpus(self):
        with self.assertRaises(ValueError):
            extract_html("<h1>Access denied</h1>", {"doc_id": "T", "url": "x"})

    def test_corpus_hash_and_review_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corpus.jsonl"
            row = asdict(fixture_doc("official"))
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "human review"):
                load_documents(path)
            row.update(review_status="human_reviewed", reviewer="UNIT TEST ONLY",
                       reviewed_at="2026-01-01T00:00:00+00:00")
            write_jsonl(path, [row])
            self.assertEqual(len(load_documents(path)), 1)
            row["paragraphs"][0]["text"] += " modified"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_documents(path)

    def test_benchmark_records_cannot_be_indexed(self):
        row = asdict(fixture_doc())
        row["reference_answer"] = "GOLD SENTINEL"
        with self.assertRaisesRegex(ValueError, "cannot be indexed"):
            Document.from_dict(row)


class ChunkAndRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.doc = fixture_doc()
        self.config = replace(Config(), tokenizer="regex_word_v1", chunk_tokens=6, overlap_tokens=2)

    def test_overlap_preserves_every_character_span(self):
        chunks = chunk_document(self.doc, 6, 2, "regex_word_v1")
        self.assertGreater(len(chunks), 2)
        hits = [{"chunk": c} for c in chunks]
        for p in self.doc.paragraphs:
            # Exact non-whitespace content is represented; boundary whitespace may be excluded.
            for i, char in enumerate(p.text):
                if not char.isspace():
                    self.assertTrue(evidence_covered({"doc_id": self.doc.doc_id,
                        "paragraph_id": p.paragraph_id, "start": i, "end": i+1,
                        "content_hash": self.doc.content_hash}, hits))

    def test_never_mix_headings(self):
        chunks = chunk_document(self.doc, 30, 2, "regex_word_v1")
        self.assertEqual(len(chunks), 2)
        self.assertNotIn("nectarine", chunks[0].text)

    def test_bm25_ranks_matching_passage(self):
        index = build_index([self.doc], self.config)
        hit = index.search("nectarine pear", 1)[0]
        self.assertEqual(hit["chunk"].heading, "Other")

    def test_dense_retrieval_normalizes_vectors(self):
        chunks = chunk_document(self.doc, 30, 2, "regex_word_v1")
        index = Index(chunks, {}, [[20, 0], [0, 3]])
        self.assertEqual(index.search("x", 1, [0, 1])[0]["chunk"].heading, "Other")
        with self.assertRaises(ValueError):
            index.search("x", 1, [0, 1, 2])

    def test_bad_vectors_are_rejected(self):
        for vector in [[0, 0], [math.nan, 1], []]:
            with self.assertRaises(ValueError):
                normalize(vector)

    def test_index_integrity_and_corpus_identity(self):
        index = build_index([self.doc], self.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "index.json"
            index.save(path)
            self.assertEqual(Index.load(path).metadata, index.metadata)
            with self.assertRaisesRegex(ValueError, "different corpus"):
                Index.load(path, "wrong-hash")
            raw = json.loads(path.read_text())
            raw["payload"]["chunks"][0]["text"] = "tampered"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "integrity"):
                Index.load(path)

    def test_partial_evidence_and_wrong_hash_do_not_count(self):
        chunk = chunk_document(self.doc, 3, 0, "regex_word_v1")[0]
        ref = {"doc_id": "TEST", "paragraph_id": "p001", "start": 0,
               "end": len(self.doc.paragraphs[0].text), "content_hash": self.doc.content_hash}
        self.assertFalse(evidence_covered(ref, [{"chunk": chunk}]))
        ref["end"] = 5
        self.assertTrue(evidence_covered(ref, [{"chunk": chunk}]))
        ref["content_hash"] = "wrong"
        self.assertFalse(evidence_covered(ref, [{"chunk": chunk}]))


class AnswerTests(unittest.TestCase):
    def test_all_claims_require_valid_citations(self):
        value = {"status": "answered", "reason": None, "message": "", "claims": [
            {"text": "A claim", "evidence_ids": ["known"]}]}
        validate_answer(value, {"known"}, True)
        for ids in [[], ["invented"], ["known", "known"]]:
            value["claims"][0]["evidence_ids"] = ids
            with self.assertRaises(ValueError):
                validate_answer(value, {"known"}, True)

    def test_uncited_answer_prose_cannot_hide_in_message(self):
        with self.assertRaises(ValueError):
            validate_answer({"status": "answered", "reason": None, "message": "UNCITED ANSWER",
                             "claims": [{"text": "ok", "evidence_ids": ["a"]}]}, {"a"}, True)

    def test_abstention_and_baseline_rules(self):
        value = {"status": "abstained", "reason": "insufficient_evidence", "message": "Missing data", "claims": []}
        validate_answer(value, set(), True)
        value["claims"] = [{"text": "hidden claim", "evidence_ids": []}]
        with self.assertRaises(ValueError):
            validate_answer(value, set(), True)
        value = {"status": "answered", "reason": None, "message": "", "claims": [{"text": "knowledge", "evidence_ids": []}]}
        validate_answer(value, set(), False)

    def test_baseline_prompt_does_not_require_empty_context_grounding(self):
        self.assertIn("MAY use your model knowledge", BASELINE_PROMPT)
        self.assertIn("Use ONLY the supplied evidence", RAG_PROMPT)

    def test_demo_false_premise_returns_original_evidence(self):
        result = demo_service().ask("A stop-limit order guarantees execution after triggering, right?", demo=True)
        self.assertEqual(result["status"], "answered")
        self.assertIn("does not guarantee", result["claims"][0]["text"])
        self.assertEqual(result["mode"], "demo_extract")
        self.assertEqual(result["estimated_usd"], 0)

    def test_demo_out_of_scope_and_missing_information(self):
        for question in ["What fee applies to my account today?", "Tell me about quantum turtles"]:
            self.assertEqual(demo_service().ask(question, demo=True)["status"], "abstained")

    def test_live_rag_cannot_use_synthetic_demo(self):
        result = demo_service().ask("What is a limit order?")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["claims"], [])

    def test_input_validation(self):
        for question in ["", "x" * 2001, None]:
            with self.assertRaises(ValueError):
                demo_service().ask(question, demo=True)


class BudgetTests(unittest.TestCase):
    def test_persistent_reservations_and_failed_usage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.json"
            ledger = Ledger(path, 1)
            first = ledger.reserve(.6, "responses")
            with self.assertRaises(BudgetError):
                Ledger(path, 1).reserve(.5, "responses")
            ledger.settle(first, .1, {"input_tokens": 10})
            ledger.reserve(.7, "embeddings")
            self.assertAlmostEqual(ledger.read()["estimated_usd"], .8)

    def test_missing_key_fails_before_network(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ProviderError, "missing"):
                OpenAIProvider(Config())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(question_id="A", family_id="f1", split="dev", question="An answerable question?",
                          label="answerable", reference_answer="Reference", required_points=["one"],
                          review_status="human_reviewed"),
                     dict(question_id="U", family_id="f2", split="dev", question="A missing fact?",
                          label="insufficient_information", reference_answer="Missing", required_points=["abstain"],
                          review_status="human_reviewed")]

    def predictions(self):
        rows = []
        for qid in ["A", "U"]:
            for arm in ["baseline", "rag"]:
                abstain = qid == "U" and arm == "rag"
                rows.append(dict(prediction_id=qid+arm, question_id=qid, mode=arm,
                    status="abstained" if abstain else "answered", message="Missing" if abstain else "",
                    claims=[] if abstain else [{"text": "Claim", "evidence_ids": ["c1"] if arm == "rag" else []}],
                    latency_ms=100, estimated_usd=.001))
        return rows

    def grades(self, predictions):
        grades = make_grade_sheet(predictions, self.rows)
        for grade in grades:
            grade.update(reviewer="UNIT TEST ONLY", reviewed_at="2026-01-01")
            grade["correctness"] = 2 if grade["prediction_id"] == "Arag" else 0
            for pair in grade["citation_support"]:
                pair["supports"] = True
        return grades

    def test_score_separates_correctness_citations_and_abstention(self):
        predictions = self.predictions()
        result = score_predictions(predictions, self.grades(predictions), self.rows)
        self.assertEqual(result["systems"]["rag"]["full_accuracy"], 1)
        self.assertEqual(result["systems"]["baseline"]["full_accuracy"], 0)
        self.assertEqual(result["systems"]["rag"]["abstention_recall"], 1)
        self.assertIsNone(result["systems"]["baseline"]["citation_support_precision"])
        self.assertEqual(result["paired_accuracy_difference"]["rag_minus_baseline"], 1)

    def test_unsupported_citation_lowers_grounded_success(self):
        predictions = self.predictions()
        grades = self.grades(predictions)
        next(g for g in grades if g["prediction_id"] == "Arag")["citation_support"][0]["supports"] = False
        report = score_predictions(predictions, grades, self.rows)
        self.assertEqual(report["systems"]["rag"]["full_accuracy"], 1)
        self.assertEqual(report["systems"]["rag"]["grounded_success"], 0)

    def test_error_is_not_a_correct_abstention(self):
        predictions = self.predictions()
        next(p for p in predictions if p["prediction_id"] == "Urag")["status"] = "error"
        report = score_predictions(predictions, self.grades(predictions), self.rows)
        self.assertEqual(report["systems"]["rag"]["abstention_recall"], 0)
        self.assertEqual(report["systems"]["rag"]["error_n"], 1)

    def test_incomplete_run_and_ungraded_data_are_rejected(self):
        predictions = self.predictions()
        with self.assertRaisesRegex(ValueError, "human grading"):
            score_predictions(predictions, make_grade_sheet(predictions, self.rows), self.rows)
        with self.assertRaisesRegex(ValueError, "Incomplete paired"):
            score_predictions(predictions[:-1], self.grades(predictions[:-1]), self.rows)

    def test_family_leakage_and_duplicate_text_are_rejected(self):
        bad = [dict(self.rows[0]), dict(self.rows[1])]
        bad[1].update(family_id="f1", split="test")
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_questions(bad)
        bad[1].update(family_id="f2", question=bad[0]["question"])
        with self.assertRaisesRegex(ValueError, "duplicate question"):
            validate_questions(bad)

    def test_human_review_and_evidence_gate(self):
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_questions(read_jsonl(ROOT / "data/eval/drafts/test.jsonl"), True)

    def test_test_requires_freeze_before_provider_is_used(self):
        rows = [dict(self.rows[0], split="test")]
        config = replace(Config(), tokenizer="regex_word_v1")
        index = build_index([fixture_doc()], config)
        def forbidden_provider(_):
            raise AssertionError("Must not call provider")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "experiment freeze"):
                run_experiment(ROOT, rows, index, [replace(config, system="baseline"), config],
                               forbidden_provider, Path(folder)/"run", "test")

    def test_drafts_have_planned_split_sizes_and_are_not_claimed_reviewed(self):
        rows = [q for s in ["smoke", "dev", "test"]
                for q in read_jsonl(ROOT / f"data/eval/drafts/{s}.jsonl")]
        result = validate_questions(rows)
        self.assertEqual(result["questions"], 80)
        self.assertEqual(result["splits"], {"smoke": 8, "dev": 32, "test": 40})
        self.assertEqual(result["reviewed"], 0)
        for row in rows:
            self.assertEqual(row["evidence_groups"], [])


if __name__ == "__main__":
    unittest.main()
