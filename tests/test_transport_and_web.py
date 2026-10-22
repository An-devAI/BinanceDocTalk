"""Mock transport and real handler execution without paid calls or listening sockets."""

from dataclasses import asdict, replace
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from binance_doctalk.answering import AnswerService
from binance_doctalk.evaluation import freeze_experiment, run_experiment, verify_freeze
from binance_doctalk.io import read_jsonl, write_jsonl
from binance_doctalk.provider import Ledger, OpenAIProvider, ProviderError
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config
from binance_doctalk.cli import ROOT
from test_pipeline import fixture_doc


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, limit):
        return self.payload[:limit]


class TransportTests(unittest.TestCase):
    def provider(self, folder):
        return OpenAIProvider(Config(), Path(folder)/"budget.json", api_key="UNIT-TEST-NOT-A-KEY")

    def test_embedding_indices_reordered_and_usage_recorded(self):
        payload = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}],
                   "model": "test-embedding", "usage": {"total_tokens": 12}}
        with tempfile.TemporaryDirectory() as folder, patch(
                "binance_doctalk.provider.urlopen", return_value=Response(payload)):
            provider = self.provider(folder)
            self.assertEqual(provider.embed(["one", "two"]), [[1, 0], [0, 1]])
            self.assertEqual(provider.usage["embedding_tokens"], 12)
            self.assertEqual(provider.ledger.read()["events"][0]["status"], "completed")

    def test_authentication_failure_is_not_retried_or_leaked(self):
        error = HTTPError("https://api.openai.com/v1/embeddings", 401, "Unauthorized", {}, None)
        with tempfile.TemporaryDirectory() as folder, patch(
                "binance_doctalk.provider.urlopen", side_effect=error) as transport:
            provider = self.provider(folder)
            with self.assertRaises(ProviderError) as caught:
                provider.embed(["test"])
            self.assertEqual(transport.call_count, 1)
            self.assertNotIn("UNIT-TEST-NOT-A-KEY", str(caught.exception))
            self.assertGreater(provider.ledger.read()["estimated_usd"], 0)

    def test_transient_failure_has_one_retry_and_retains_unknown_reserve(self):
        error = HTTPError("https://api.openai.com/v1/embeddings", 429, "Limited", {}, None)
        payload = {"data": [{"index": 0, "embedding": [1, 0]}], "usage": {"total_tokens": 2}}
        with (tempfile.TemporaryDirectory() as folder,
              patch("binance_doctalk.provider.urlopen",
                    side_effect=[error, Response(payload)]) as transport,
              patch("binance_doctalk.provider.time.sleep")):
            provider = self.provider(folder)
            provider.embed(["test"])
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(provider.retry_count, 1)
            self.assertEqual(provider.ledger.read()["events"][0]["status"], "reserved_usage_unknown")

    def test_structured_generation_request_and_provider_refusal(self):
        answer = {"status": "answered", "reason": None, "message": "",
                  "claims": [{"text": "Test fact", "evidence_ids": []}]}
        payload = {"status": "completed", "model": "test-model",
                   "usage": {"input_tokens": 20, "output_tokens": 10},
                   "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]}
        with tempfile.TemporaryDirectory() as folder, patch(
                "binance_doctalk.provider.urlopen", return_value=Response(payload)) as transport:
            provider = self.provider(folder)
            self.assertEqual(provider.generate("instructions", "question", []), answer)
            sent = json.loads(transport.call_args.args[0].data)
            self.assertFalse(sent["store"])
            self.assertEqual(sent["text"]["format"]["type"], "json_schema")
            self.assertEqual(provider.usage["output_tokens"], 10)
            payload["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
            transport.return_value = Response(payload)
            with self.assertRaisesRegex(ProviderError, "not an evidence abstention"):
                provider.generate("instructions", "question", [])

    def test_invalid_generated_citation_becomes_error_with_usage_preserved(self):
        config = replace(Config(), tokenizer="regex_word_v1")
        class Embedder:
            def embed(self, texts, *, purpose="document"): return [[1, 0] for _ in texts]
        index = build_index([fixture_doc("official")], config, Embedder(), reviewed=True)
        answer = {"status": "answered", "reason": None, "message": "", "claims": [
            {"text": "Unvalidated model text", "evidence_ids": ["INVENTED"]}]}
        payloads = [Response({"data": [{"index": 0, "embedding": [1, 0]}], "usage": {"total_tokens": 3}}),
                    Response({"status": "completed", "usage": {"input_tokens": 10, "output_tokens": 5},
                              "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer)}]}]})]
        with tempfile.TemporaryDirectory() as folder, patch(
                "binance_doctalk.provider.urlopen", side_effect=payloads):
            result = AnswerService(index, config, self.provider(folder)).ask("alpha beta")
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["claims"], [])
            self.assertEqual(result["usage"]["output_tokens"], 5)


class WebHandlerTests(unittest.TestCase):
    def get_handler(self):
        from binance_doctalk.web import serve
        captured = []
        class Server:
            def __init__(self, address, handler): captured.append(handler)
            def serve_forever(self): pass
            def server_close(self): pass
        with patch("binance_doctalk.web.HTTPServer", Server), patch("builtins.print"):
            serve()
        return captured[0]

    def request(self, method, path, body=None, host="127.0.0.1:8765", origin=None):
        handler = self.get_handler()
        raw_body = json.dumps(body).encode() if body is not None else b""
        headers = f"{method} {path} HTTP/1.0\r\nHost: {host}\r\nContent-Length: {len(raw_body)}\r\n"
        if origin: headers += f"Origin: {origin}\r\n"
        raw = headers.encode() + b"\r\n" + raw_body
        class Connection:
            def __init__(self): self.output = b""
            def makefile(self, *args): return BytesIO(raw)
            def sendall(self, data): self.output += data
        connection = Connection()
        handler(connection, ("127.0.0.1", 123), object())
        response_headers, response = connection.output.split(b"\r\n\r\n", 1)
        return response_headers, response

    def test_home_and_status(self):
        headers, body = self.request("GET", "/")
        self.assertIn(b"200 OK", headers)
        self.assertIn(b"Inspect the evidence", body)
        _, body = self.request("GET", "/api/status")
        self.assertEqual(json.loads(body)["quality_results"], "not_measured")

    def test_question_roundtrip_includes_resolved_source(self):
        headers, body = self.request("POST", "/api/ask", {"question": "Does a stop-limit guarantee execution?", "mode": "demo"})
        result = json.loads(body)
        self.assertIn(b"200 OK", headers)
        self.assertEqual(result["status"], "answered")
        self.assertIn("does not guarantee", result["claims"][0]["text"])
        self.assertIn("text", result["retrieved"][0])

    def test_csrf_and_dns_rebinding_guards(self):
        headers, _ = self.request("GET", "/api/status", host="attacker.test:8765")
        self.assertIn(b"403", headers)
        headers, _ = self.request("POST", "/api/ask", {"question": "test"}, origin="https://attacker.test")
        self.assertIn(b"403", headers)


class FreezeTests(unittest.TestCase):
    def test_frozen_test_is_consumed_once_and_files_cannot_change(self):
        config = replace(Config(), tokenizer="regex_word_v1")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            class FakeProvider:
                def __init__(self, configuration):
                    self.config = configuration
                    self.usage = {}
                    self.retry_count = 0
                    self.actual_models = set()
                    self.ledger = Ledger(folder/"ledger.json", 1)
                def embed(self, texts, *, purpose="document"): return [[1, 0] for _ in texts]
                def generate(self, instructions, question, context):
                    return {"status": "abstained", "reason": "insufficient_evidence",
                            "message": "UNIT-TEST OUTPUT, not a quality measurement", "claims": []}
            corpus_path, index_path = folder/"corpus.jsonl", folder/"index.json"
            doc = fixture_doc("official")
            write_jsonl(corpus_path, [{**asdict(doc), "review_status": "human_reviewed",
                                      "reviewer": "UNIT TEST ONLY", "reviewed_at": "2026-01-01"}])
            index = build_index([doc], config, FakeProvider(config), reviewed=True)
            index.save(index_path)
            question_paths = []
            for split in ["smoke", "dev", "test"]:
                rows = read_jsonl(ROOT/f"data/eval/drafts/{split}.jsonl")
                for row in rows:
                    row.update(review_status="human_reviewed", reviewer="UNIT TEST FIXTURE ONLY",
                               reviewed_at="2026-01-01")
                    if row["label"] == "answerable":
                        row["evidence_groups"] = [[{"doc_id": doc.doc_id, "paragraph_id": "p001",
                            "start": 0, "end": 5, "content_hash": doc.content_hash}]]
                path = folder/f"{split}.jsonl"
                write_jsonl(path, rows)
                question_paths.append(path)
            baseline_path, rag_path = folder/"baseline.toml", folder/"rag.toml"
            for path, arm in [(baseline_path, "baseline"), (rag_path, "rag")]:
                path.write_text(f'system = "{arm}"\ntokenizer = "regex_word_v1"\n')
            freeze_path = folder/"freeze.json"
            frozen = freeze_experiment(ROOT, corpus_path, index_path, question_paths,
                                       [baseline_path, rag_path], freeze_path)
            self.assertEqual(len(frozen["test_question_ids"]), 40)
            configs = [Config.load(baseline_path), Config.load(rag_path)]
            rows = read_jsonl(question_paths[-1])
            result = run_experiment(ROOT, rows, index, configs, FakeProvider,
                                    folder/"run1", "test", freeze_path)
            self.assertEqual(result["prediction_count"], 80)
            with self.assertRaisesRegex(ValueError, "already consumed"):
                run_experiment(ROOT, rows, index, configs, FakeProvider,
                               folder/"run2", "test", freeze_path)
            rows[0]["question"] = "Tampered question with the same ID"
            with self.assertRaisesRegex(ValueError, "differ from frozen"):
                run_experiment(ROOT, rows, index, configs, FakeProvider,
                               folder/"run3", "test", freeze_path)
            rag_path.write_text('system = "rag"\ntop_k = 5\n')
            with self.assertRaisesRegex(ValueError, "Frozen input"):
                verify_freeze(ROOT, freeze_path)
