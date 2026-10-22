# Local Hugging Face inference on Kaggle

The optional backend uses Qwen3-4B-Instruct-2507 (FP16, GPU 0) and BGE-small-en-v1.5
(CPU, normalized dense vectors). Both revisions are pinned in `configs/kaggle-rag.toml`.
No fine-tuning, paid API, reranker, quantization, or serving framework is involved.

```bash
python3 scripts/kaggle_task.py prepare --gpu --accelerator NvidiaTeslaT4 --folder artifacts/local/kaggle/gpu
python3 scripts/kaggle_task.py push --folder artifacts/local/kaggle/gpu
python3 scripts/kaggle_task.py status --folder artifacts/local/kaggle/gpu
python3 scripts/kaggle_task.py output --folder artifacts/local/kaggle/gpu --download-to artifacts/local/kaggle/gpu-output --file-pattern '(^|/)(gpu-smoke.json|core-tests.txt|benchmark-audit.json)$'
```

The notebook keeps model caches in `/kaggle/temp`, outside saved outputs. Installation uses the
Kaggle image's PyTorch; actual torch/CUDA/GPU versions are recorded. GPU access/quota is a
precondition, not guaranteed by the project. No automatic paid fallback is used.

The GPU smoke uses original synthetic notes and three integration questions with both generation
arms. It validates real inference and JSON/citation structure, not semantic accuracy or official
Binance answer quality. It does not consume development or held-out questions.

Real BGE integration completed on a private Kaggle CPU notebook; see
[`kaggle_embedding_run.json`](kaggle_embedding_run.json). Qwen GPU inference is tracked separately.
To reproduce the CPU check, prepare with `--embedding-smoke` instead of `--gpu` and use a separate folder.
Push commands cap execution at 3,600 seconds.

For official data, use `build`, `ask`, `retrieval-eval`, `freeze`, and `evaluate` with
`--config configs/kaggle-rag.toml` or the paired `--baseline-config configs/kaggle-baseline.toml`
and `--rag-config configs/kaggle-rag.toml`. Existing source/gold review gates still apply.
Change `top_k` in a separate development configuration for k=5; never edit the chosen config after freeze.

BGE tokenizer windows preserve paragraph character offsets. Passage inputs over 512 tokens and
prompts over 4096 tokens are rejected rather than silently truncated. Qwen uses greedy decoding,
768 maximum output tokens, and a fixed seed; exact GPU reproducibility is not guaranteed.
JSON Schema token constraints are applied during decoding using LM Format Enforcer 0.11.3.
The local schema separates answered and abstained objects and restricts citation IDs to supplied
chunks. The shared validator still rejects invalid/truncated output and duplicate citations;
there is no automatic output repair. Grammar validity does not establish factual correctness.
Models are process-cached so paired baseline/RAG providers share GPU weights.

`estimated_usd` retains the application's existing **API-only** meaning: zero for this backend.
`runtime.compute_usd` is null because GPU time is not priced. Warmup/model setup is recorded
separately and excluded from per-query timing. All raw private outputs remain ignored by Git.

Model cards: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 and
https://huggingface.co/BAAI/bge-small-en-v1.5. Their licenses are separate from project code and Binance content.

## Prepared development configurations

The k=3 pair is `configs/kaggle-baseline.toml` / `configs/kaggle-rag.toml`; the k=5 pair is
`configs/kaggle-k5-baseline.toml` / `configs/kaggle-k5-rag.toml`. Within a pair only `system` differs.
The index is reusable across k because embedding/chunk settings are identical. Choose using the
predeclared development criterion in `docs/evaluation.md`; freeze only the chosen pair.

`python3 scripts/audit_splits.py --output artifacts/local/split-audit.json` produces a ranked
shortlist of cross-split lexical similarities for semantic inspection. Scores below a threshold
are not evidence of independence. No questions or labels are automatically changed.

## Formatting failure and correction

The first GPU integration completed actual Qwen inference but failed strict validation on four
of six original-note outputs: trailing JSON content, missing fields, and an empty abstention
message. The failed outputs are preserved; no held-out questions were involved. Output lengths
were 19–123 tokens, below the 768-token limit, ruling out output truncation for these failures.

`python3 scripts/check_format_grammar.py` replays the four actual malformed outputs against the
decoding grammar, accepts valid answered/abstained examples, and blocks invented citation IDs.
This runs with `lm-format-enforcer` installed and requires neither GPU nor LLM weights.

Both baseline and RAG use the same constrained decoding mechanism with their respective citation
contracts. Report structural citation validity as constrained by construction, separately from
human semantic support and answer accuracy. This dependency was added to address measured
formatting failures, not as an additional serving framework.

The helper also accepts `--accelerator NvidiaL4`. The actual GPU name, CUDA and available VRAM
reported by the runtime are authoritative; the platform may not allocate the requested shape.
A dedicated Kaggle session is used: pinning Transformers may conflict with unused Gradio/Diffusers
packages bundled in the base image. Those packages are not part of this application or its checks.

Primary integration reference: https://github.com/noamgat/lm-format-enforcer.
