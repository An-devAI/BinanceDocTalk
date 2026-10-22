"""Loopback-only, dependency-free UI; production hosting is intentionally separate."""

from __future__ import annotations

from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from .answering import AnswerService
from .cli import benchmark_status, demo_service
from .provider import OpenAIProvider
from .retrieval import Index
from .schemas import Config


def serve(port: int = 8765, index_path: Path | None = None,
          config_path: Path | None = None, report_path: Path | None = None) -> None:
    demo = demo_service()
    config = Config.load(config_path) if config_path else Config()
    live_index = Index.load(index_path) if index_path and index_path.exists() else None
    report = json.loads(report_path.read_text()) if report_path else None
    if report is not None and (report.get("status") != "human_graded" or "systems" not in report):
        raise ValueError("UI report must be a completed human-graded metrics file")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # No question text or provider payloads in server logs.
            pass

        def send(self, data, code=200, content_type="application/json"):
            body = json.dumps(data).encode() if content_type == "application/json" else data
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def trusted_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.trusted_host():
                return self.send({"error": "Loopback host required"}, 403)
            path = urlparse(self.path).path
            if path == "/":
                return self.send(Path(__file__).with_name("ui.html").read_bytes(),
                                 content_type="text/html")
            if path == "/api/status":
                return self.send({"api_key_configured": bool(os.getenv("OPENAI_API_KEY")),
                                  "live_index_loaded": live_index is not None,
                                  "demo": demo.index.metadata, "model": config.model,
                                  "quality_results": "saved_report_available" if report else "not_measured",
                                  "benchmark": benchmark_status()})
            if path == "/api/report":
                return self.send(report if report else {"status": "not_measured"})
            return self.send({"error": "Not found"}, 404)

        def do_POST(self):
            if not self.trusted_host():
                return self.send({"error": "Loopback host required"}, 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                return self.send({"error": "Cross-origin requests are not accepted"}, 403)
            if self.path != "/api/ask":
                return self.send({"error": "Not found"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 12_000:
                    raise ValueError("Request body must be at most 12 KB")
                payload = json.loads(self.rfile.read(length))
                mode = payload.get("mode", "demo")
                if mode not in {"demo", "rag", "baseline"}:
                    raise ValueError("Unknown mode")
                if mode == "demo":
                    service = demo
                else:
                    selected = replace(config, system=mode)
                    service = AnswerService(live_index, selected, OpenAIProvider(selected))
                result = service.ask(payload.get("question", ""), demo=mode == "demo")
                # Local UI can inspect context. Evaluation artifacts deliberately omit raw context.
                for hit in result["retrieved"]:
                    hit["text"] = service.source_text(hit["chunk_id"])
                return self.send(result)
            except (ValueError, RuntimeError, TypeError) as exc:
                return self.send({"status": "error", "message": str(exc)}, 400)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"BinanceDocTalk running at http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
