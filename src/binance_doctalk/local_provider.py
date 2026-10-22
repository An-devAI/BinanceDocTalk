"""Optional, pinned Hugging Face inference. Importing this module needs no ML packages.

Weights are shared across provider instances; accounting remains per experiment arm.
No network API, quantization, automatic output repair, or CPU generation fallback.
"""
from __future__ import annotations

from functools import lru_cache
import json
import threading
import time

from .chunking import BGE_REVISION
from .provider import ProviderError
from .schemas import Config

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_RUNTIME_LOCK = threading.RLock()


def local_answer_schema(grounded: bool, allowed_ids: list[str]) -> dict:
    """Encode the existing validator contract without deciding factual correctness."""
    abstained = {"type": "object", "additionalProperties": False, "properties": {
        "status": {"type": "string", "enum": ["abstained"]},
        "reason": {"type": "string", "enum": ["insufficient_evidence", "out_of_scope"]},
        "message": {"type": "string", "minLength": 1, "maxLength": 400},
        "claims": {"type": "array", "maxItems": 0, "items": {"type": "string"}},
    }, "required": ["status", "reason", "message", "claims"]}
    if grounded and not allowed_ids:
        return abstained
    citations = {"type": "array", "items": {"type": "string"}, "maxItems": 0}
    if grounded:
        citations = {"type": "array", "minItems": 1, "maxItems": len(allowed_ids),
                     "items": {"type": "string", "enum": sorted(set(allowed_ids))}}
    answered = {"type": "object", "additionalProperties": False, "properties": {
        "status": {"type": "string", "enum": ["answered"]},
        "reason": {"type": "null"}, "message": {"type": "string", "enum": [""]},
        "claims": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
            "type": "object", "additionalProperties": False, "properties": {
                "text": {"type": "string", "minLength": 1, "maxLength": 800},
                "evidence_ids": citations}, "required": ["text", "evidence_ids"]}},
    }, "required": ["status", "reason", "message", "claims"]}
    return {"anyOf": [answered, abstained]}


@lru_cache(maxsize=1)
def constraint_tokenizer_data(model: str, revision: str):
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    tokenizer = generation_runtime(model, revision)[1]
    return build_token_enforcer_tokenizer_data(tokenizer)


def build_prefix_enforcer(config: Config, schema: dict):
    from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    parser = JsonSchemaParser(schema, config=CharacterLevelParserConfig(force_json_field_order=True))
    return build_transformers_prefix_allowed_tokens_fn(
        constraint_tokenizer_data(config.model, config.model_revision), parser)


class NoApiLedger:
    """Zero API spend only; GPU compute is explicitly not priced by this adapter."""
    def read(self):
        return {"estimated_usd": 0.0}


def gpu_preflight(torch) -> dict:
    if not torch.cuda.is_available():
        raise ProviderError("CUDA GPU unavailable; enable a Kaggle GPU session with quota")
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if total < 15_000_000_000 or free < 11_000_000_000:
        raise ProviderError("Local generation requires a 16 GB class GPU with at least 11 GB free VRAM")
    return {"gpu": device.name, "total_vram_bytes": total, "free_vram_at_load_bytes": free,
            "torch": torch.__version__, "cuda": torch.version.cuda, "dtype": "float16"}


@lru_cache(maxsize=1)
def embedding_runtime(model: str, revision: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model, revision=revision, device="cpu", trust_remote_code=False)


@lru_cache(maxsize=1)
def generation_runtime(model: str, revision: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hardware = gpu_preflight(torch)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
    network = AutoModelForCausalLM.from_pretrained(
        model, revision=revision, torch_dtype=torch.float16, trust_remote_code=False,
        use_safetensors=True, attn_implementation="sdpa").to("cuda:0").eval()
    torch.cuda.synchronize()
    hardware["model_load_seconds"] = time.perf_counter() - started
    return torch, tokenizer, network, hardware


class LocalProvider:
    def __init__(self, config: Config):
        if config.embedding_revision != BGE_REVISION:
            raise ProviderError("BGE revision differs from the pinned chunk tokenizer")
        self.config = config
        self.ledger = NoApiLedger()
        self.retry_count = 0
        self.usage = {"input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        self.actual_models: set[str] = set()
        self.last_raw_output = ""
        self.hardware: dict = {}

    def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        if purpose not in {"query", "document"}:
            raise ValueError("Unknown embedding purpose")
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Cannot embed empty input")
        with _RUNTIME_LOCK:
            encoder = embedding_runtime(self.config.embedding_model, self.config.embedding_revision)
            inputs = [QUERY_PREFIX + text if purpose == "query" else text for text in texts]
            lengths = [len(encoder.tokenizer(text, truncation=False)["input_ids"]) for text in inputs]
            if any(length > encoder.max_seq_length for length in lengths):
                raise ProviderError("Embedding input exceeds model limit; silent truncation is disabled")
            vectors = encoder.encode(inputs, batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
            self.usage["embedding_tokens"] += sum(lengths)
            self.actual_models.add(self.config.embedding_model + "@" + self.config.embedding_revision)
            return vectors.tolist()

    def generate(self, instructions: str, question: str, context: list[dict]) -> dict:
        with _RUNTIME_LOCK:
            self.last_raw_output = ""
            torch, tokenizer, model, self.hardware = generation_runtime(
                self.config.model, self.config.model_revision)
            schema_object = local_answer_schema(self.config.system == "rag",
                                                [c["chunk_id"] for c in context])
            prefix_enforcer = build_prefix_enforcer(self.config, schema_object)
            schema = json.dumps(schema_object, separators=(",", ":"))
            messages = [
                {"role": "system", "content": instructions + "\nReturn ONLY one JSON object, no Markdown. "
                 "Use this JSON schema: " + schema},
                {"role": "user", "content": json.dumps({"question": question, "evidence": context},
                                                         ensure_ascii=False)},
            ]
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt")
            count = inputs["input_ids"].shape[-1]
            if count > self.config.max_input_tokens:
                raise ProviderError("Prompt exceeds max_input_tokens; evidence was not truncated")
            inputs = inputs.to("cuda:0")
            torch.manual_seed(self.config.seed)
            torch.cuda.synchronize()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=self.config.max_output_tokens,
                                        pad_token_id=tokenizer.eos_token_id,
                                        prefix_allowed_tokens_fn=prefix_enforcer)
            torch.cuda.synchronize()
            new_tokens = output[0][count:]
            self.usage["input_tokens"] += count
            self.usage["output_tokens"] += len(new_tokens)
            self.actual_models.add(self.config.model + "@" + self.config.model_revision)
            self.hardware["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            self.last_raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            if len(new_tokens) >= self.config.max_output_tokens and int(new_tokens[-1]) != tokenizer.eos_token_id:
                raise ProviderError("Generation reached output limit; incomplete output was not used")
            try:
                return json.loads(self.last_raw_output)
            except json.JSONDecodeError:
                raise ProviderError("Local model returned invalid JSON; output was not repaired") from None

    def warmup(self):
        """Load and warm the shared models before timed evaluation or held-out consumption."""
        self.embed(["market order"], purpose="query")
        self.generate("Answer a formatting connectivity check, not a factual question.", "Return an abstention.", [])
        return self.hardware

    def runtime_info(self):
        return {"provider": "huggingface_local", "model_revision": self.config.model_revision,
                "embedding_revision": self.config.embedding_revision, "decoding": "greedy",
                "structured_decoding": "lm-format-enforcer/json-schema/strict-field-order",
                "compute_usd": None, "cost_basis": "API only; GPU compute not priced", **self.hardware}
