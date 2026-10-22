"""OpenAI REST adapter with strict output and a persistent conservative budget ledger.

Uses the documented REST API so the core can run without installing an SDK.
Never includes provider response bodies, credentials, or request text in error messages.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .io import utc_now, write_json
from .schemas import ANSWER_SCHEMA, Config


class ProviderError(RuntimeError):
    pass


class BudgetError(ProviderError):
    pass


class Ledger:
    """Atomically reserve worst-case costs before each request, including retries.

    Failed requests retain reservations because their billable usage is unknown.
    Prices are configuration estimates, not a guarantee of the provider's invoice.
    """

    def __init__(self, path: Path, ceiling: float):
        self.path, self.ceiling = path, ceiling

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_suffix(".lock")
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise BudgetError("Budget ledger is in use; inspect a stale lock after a crash") from exc
        try:
            os.close(fd)
            yield
        finally:
            lock.unlink()

    def read(self) -> dict:
        if not self.path.exists():
            return {"estimated_usd": 0.0, "events": []}
        return json.loads(self.path.read_text())

    def reserve(self, cost: float, endpoint: str) -> int:
        with self.locked():
            state = self.read()
            if state["estimated_usd"] + cost > self.ceiling:
                raise BudgetError("Local estimated API budget would be exceeded")
            event = {"endpoint": endpoint, "reserved_usd": cost, "charged_estimate_usd": cost,
                     "status": "reserved_usage_unknown", "created_at": utc_now()}
            state["events"].append(event)
            state["estimated_usd"] += cost
            write_json(self.path, state)
            return len(state["events"]) - 1

    def settle(self, event_id: int, cost: float, usage: dict) -> None:
        with self.locked():
            state = self.read()
            event = state["events"][event_id]
            state["estimated_usd"] += cost - event["charged_estimate_usd"]
            event.update(status="completed", charged_estimate_usd=cost, usage=usage)
            write_json(self.path, state)


class OpenAIProvider:
    def __init__(self, config: Config, ledger_path: Path = Path("artifacts/local/budget.json"),
                 api_key: str | None = None):
        self.config = config
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.key:
            raise ProviderError("OPENAI_API_KEY is missing. Configure .env locally or use demo mode.")
        self.ledger = Ledger(ledger_path, config.budget_usd)
        self.retry_count = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        self.actual_models: set[str] = set()

    def _post(self, endpoint: str, payload: dict, reserve: float) -> tuple[dict, int]:
        raw = json.dumps(payload).encode()
        for attempt in range(self.config.max_retries + 1):
            event = self.ledger.reserve(reserve, endpoint)
            request = Request("https://api.openai.com/v1/" + endpoint, data=raw,
                              headers={"Authorization": f"Bearer {self.key}",
                                       "Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=self.config.timeout_seconds) as response:
                    body = response.read(8_000_001)
                    if len(body) > 8_000_000:
                        raise ProviderError("Provider response exceeded size limit")
                    return json.loads(body), event
            except HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt == self.config.max_retries:
                    raise ProviderError(f"Provider HTTP {exc.code}; no response body logged") from None
            except (URLError, TimeoutError, OSError):
                if attempt == self.config.max_retries:
                    raise ProviderError("Provider network/timeout failure") from None
            except (json.JSONDecodeError, UnicodeError):
                raise ProviderError("Provider returned malformed JSON") from None
            self.retry_count += 1
            time.sleep(1)
        raise ProviderError("Request failed")

    def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if purpose not in {"query", "document"}:
            raise ValueError("Unknown embedding purpose")
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty input")
        config = self.config
        estimate = (sum(len(t.encode()) for t in texts) + 4096) * config.embedding_usd_per_million / 1e6
        response, event = self._post("embeddings", {
            "model": config.embedding_model, "input": texts, "encoding_format": "float"}, estimate)
        rows = response.get("data", [])
        if len(rows) != len(texts) or sorted(r["index"] for r in rows) != list(range(len(texts))):
            raise ProviderError("Embedding response indices do not match request")
        count = response["usage"]["total_tokens"]
        self.usage["embedding_tokens"] += count
        self.actual_models.add(response.get("model", config.embedding_model))
        self.ledger.settle(event, count * config.embedding_usd_per_million / 1e6,
                           {"embedding_tokens": count})
        return [row["embedding"] for row in sorted(rows, key=lambda r: r["index"])]

    def generate(self, instructions: str, question: str, context: list[dict]) -> dict:
        config = self.config
        payload = {
            "model": config.model, "instructions": instructions,
            "input": json.dumps({"question": question, "evidence": context}, ensure_ascii=False),
            "reasoning": {"effort": config.reasoning_effort},
            "max_output_tokens": config.max_output_tokens,
            "store": False,
            "text": {"format": {"type": "json_schema", "name": "grounded_answer",
                                  "strict": True, "schema": ANSWER_SCHEMA}},
        }
        estimate_input = len(json.dumps(payload).encode()) + 4096
        reserve = (estimate_input * config.input_usd_per_million
                   + config.max_output_tokens * config.output_usd_per_million) / 1e6
        response, event = self._post("responses", payload, reserve)
        usage = response.get("usage")
        if usage is not None:
            counts = {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
            for name, count in counts.items():
                self.usage[name] += count
            self.ledger.settle(event, (counts["input_tokens"] * config.input_usd_per_million
                                      + counts["output_tokens"] * config.output_usd_per_million) / 1e6,
                               counts)
        self.actual_models.add(response.get("model", config.model))
        if response.get("status") != "completed":
            raise ProviderError("Generation did not complete; output was not used")
        texts = []
        for item in response.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        raise ProviderError("Provider safety refusal; not an evidence abstention")
                    if part.get("type") == "output_text":
                        texts.append(part["text"])
        try:
            return json.loads("".join(texts))
        except json.JSONDecodeError:
            raise ProviderError("No parseable structured model output") from None


def create_provider(config: Config, ledger_path: Path = Path("artifacts/local/budget.json")):
    if config.provider == "huggingface_local":
        from .local_provider import LocalProvider
        return LocalProvider(config)
    return OpenAIProvider(config, ledger_path)
