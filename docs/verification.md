# Verification record — updated 2026-10-23 (UTC+07)

## Completed locally

| Check | Observed result |
|---|---|
| Core and integration suite | 97 tests pass locally on Python 3.12.3 and on the official-source Kaggle smoke snapshot |
| Benchmark v2 structural audit | 92 drafts; splits 10/36/46; labels 56 answerable / 12 candidate gaps / 24 out-of-scope; zero approvals |
| CLI demonstration | Actual original-note retrieval, false-premise example and out-of-scope response |
| API transport | Mocked structured generation, embedding order, usage, 401/no retry, 429/one retry |
| Held-out controls | Review/audit/freeze gates, payload integrity, question-set identity, renamed-freeze rejection and consumed-family guard tested on synthetic fixtures |
| UI server handlers | In-memory handler tests plus actual loopback server startup: HTTP 200, 12,043-byte page |
| Browser JavaScript | Syntax checked with Node.js; not browser interaction testing |
| Python package | Wheel builds without fetching runtime dependencies; HTML asset included |
| Core lockfile | `uv lock --check --offline` passes; no runtime dependencies |
| Kaggle GPU notebook | COMPLETE; 67 tests pass on Python 3.13.15; six actual Qwen outputs pass structure/citation checks |
| Demo videos | Original 27-second CLI preview plus 30-second actual-Qwen output presentation; VP8 WebM, 1280×720 |

The test suite uses original/synthetic fixtures and mocked network responses. Its numerical
assertions validate the metric implementation; they are not project accuracy or citation scores.

## Not verified / not completed

- No actual OpenAI call: no OpenAI key is configured in the implementation environment.
- Direct HTTP collection failed for 20 URLs. A subsequent web-reader import retrieved 20 snapshots
  containing 481 paragraphs. These may be cached and remain pending human review; no reviewed
  official corpus is claimed. See `docs/collection_workflow.md`.
- Owner chat signoff covers 20 sources and 10 smoke references. Dev/test gold, official held-out
  inference and semantic output grading remain incomplete.
- Streamlit/tiktoken were installed and exercised with AppTest and real tokenizer calls;
  see `docs/local_ui_run.json`. No live browser visual QA is claimed.
- No real browser UI inspection: browser control reported no available browser, and local browser
  process startup was blocked. The poster is a rendered CLI presentation, not a browser screenshot.
- Online hosting has not been performed. GitHub Actions passed core checks on Python 3.11–3.13,
  optional Streamlit checks and actual grammar replay for `f6fa78f`:
  https://github.com/An-devAI/BinanceDocTalk/actions/runs/39825385108.

## V2 revision verification

Corpus curation preserves the 481-paragraph parent and produces 477 selected paragraphs across
20 sources. Seven operations remove four paragraphs and trim three; paragraph IDs are retained.
Public metadata, raw snapshot date provenance, edit hashes and known source conflicts are recorded.
The corpus and all 92 questions remain pending human review. Twelve new gap audits enumerate every
source, but their lexical findings are not a claim of human-verified insufficiency.

New tests exercise independent labels/types, scope policy bindings, whole-corpus audit coverage,
stale hashes, evidence groups per fact, refusal reason confusion, injection behavior, source aliases,
protected exception chunks and renamed held-out freeze rejection. Source and benchmark v1 artifacts
are preserved. The complete local packet is `artifacts/local/review-pack-v3/`.
Optional Streamlit AppTest and real grammar replay passed again locally after this revision.
The allowlisted archive reproduced all 79 tests and the 92-question audit in a clean temporary
extraction. All 518 local review-packet links resolve. Protected source groups were exercised
with cl100k tokenization (139, 125 and 196 tokens); actual BGE tokenization on these new source
groups remains unverified locally. The pinned BGE path validates its real token budget at build time.
No additional GPU inference or official held-out run was performed for v2. The GPU counts below
refer to their exact historical submitted snapshots, not the updated tree.

## Kaggle account and execution state

Kaggle CLI 2.2.4 was installed in `.venv-kaggle` after network access became available.
Existing OAuth was refreshed using the official SDK. Tokens were not printed, embedded in source,
or copied into public artifacts. The helper preserves automatic refresh by removing an incomplete
legacy username-only authentication configuration from CLI child environments.

Actual remote run on 2026-10-22:

- Notebook: [private CPU smoke run](https://www.kaggle.com/code/annguyencherry/binancedoctalk-smoke-000bb9cf), version 1.
- Observed status: `KernelWorkerStatus.COMPLETE`.
- Source SHA-256: `000bb9cf5e07ff3437df664ced969fec99ff26860e48792832cf5b1210c6b177`.
- Runtime: Python 3.13.15; 45 tests passed in 0.238 seconds (test duration, not total job time).
- Three original-note queries executed: market/limit explanation, stop-limit false-premise correction,
  and abstention on the current account fee question.
- Benchmark audit: 80 drafts, split 8/32/40, 56 answerable / 24 insufficient, zero human-reviewed.
- Paid API probe disabled; no OpenAI model-quality metrics measured.

Downloaded run logs and outputs are under ignored `artifacts/local/kaggle/`. A sanitized run
summary is in `docs/kaggle_run.json`. The submitted snapshot predates documentation updates and a
slug-generation correction; its hash identifies that exact executed source, not the current tree.
This is a successful infrastructure smoke run, not the official held-out study.

## Real embedding integration on Kaggle CPU

Private run `annguyencherry/binancedoctalk-embedding-a52f301e` completed on Python 3.13.15.
All 61 tests in its submitted snapshot passed. Actual BGE inference produced 384-dimensional
normalized vectors for 12 original-note chunks; paragraph spans were verified. Two query embedding
plus retrieval calls took 32.85 ms and 26.42 ms after setup. Setup took 54.36 seconds; these two
observations are integration timings, not a representative latency benchmark or answer-quality score.
Pinned Transformers 4.57.6 and Sentence Transformers 5.2.0 worked with torch 2.11.0+cpu.
See `docs/kaggle_embedding_run.json` for actual outputs and source hash.

## Real Qwen GPU integration — completed

[Private executed notebook](https://www.kaggle.com/code/annguyencherry/binancedoctalk-gpu-2918daa9-l4),
version 1, reports `KernelWorkerStatus.COMPLETE`. Remote metadata confirms private visibility.
The executed source SHA-256 is `2918daa924cb46d82cd2a2d3a464bde3d1fb1fb1f2130af77c42c2589f40f58c`.

- Runtime: Python 3.13.15; Tesla T4; torch 2.11.0+cu128; CUDA 12.8; FP16.
- The requested shape was `NvidiaL4`, but device telemetry reported Tesla T4. Report the observed device.
- Qwen and BGE revisions are pinned in the saved configuration; shared model weights serve both arms.
- All 67 tests passed remotely. Three original-note questions produced six baseline/RAG outputs,
  with zero structural validation errors. Both current-account-fee queries abstained.
- Setup including model loading/warmup took 92.23 seconds. Observed per-query times were approximately
  2.00–14.92 seconds; six illustrative outputs do not establish a representative latency benchmark.
- PyTorch peak allocated VRAM was 8,581,229,056 bytes; this is not total driver/device allocation.
- API spend was zero. Compute cost is unpriced; no paid fallback or pay-to-scale change was made.

The two initial runs each failed four of six outputs; `kaggle_gpu_failures.json` preserves those
results. Token-level JSON constraints fixed syntax/field failures. An intermediate structurally
valid run exposed factual/nonsensical prose in abstention messages; fixed limitation templates
closed that contract loophole. The shared validator was not weakened and outputs were not repaired.

See `kaggle_gpu_run.json`, `demo/gpu-recorded-results.json`, the recorded HTML preview and 30-second
video. All six final outputs are retained. Structural validity is constrained by construction;
correctness, refusal classification and semantic citation support are **not human-graded**.
Observed limitations include a potential market-order guarantee overstatement and a baseline
refusal of an in-scope question. Human source/gold review remains pending by owner choice;
no held-out inference, quality metric, or claimed RAG improvement exists.

## Reproduce local checks

```bash
python3 -m unittest discover -s tests -v
python3 scripts/doctalk.py audit-benchmark
python3 scripts/doctalk.py demo
python3 scripts/kaggle_task.py doctor --check-auth
```

Raw local check logs, budget/configuration state and remote outputs belong under ignored
`artifacts/local/`. Inspect any future public results against the frozen manifest and human grades.

## Optional Hugging Face import

Pinned full Parquet artifacts were downloaded and SHA-256/schema/count checked locally.
Observed: 3,487 corpus rows (581 Binance); 497 QA (84 Binance from 42 URLs). Selection stages
15 source candidates (nine URL matches/six supplemental) and 12 development QA candidates.
Public metadata: `huggingface_import_v1.json`; actual source/QA and full comparison pages stay
under ignored local artifacts. All candidate labels/types/evidence remain unassigned and pending.
The active corpus/92-case benchmark hashes are preserved; no held-out model outputs were read.

Ten additional core tests exercise external import boundaries. The optional real Parquet fixture
check passes with Arrow 26.0.0; core runtime dependencies remain empty. These checks and source
counts are not human gold review or LLM quality results. Historical GPU snapshots below retain
their original test counts and source hashes.

The latest allowlisted source archive reproduces all 89 core tests and the unchanged 92-case
audit in a clean temporary extraction. The HF packet's 43 links resolve; public metadata contains
no copied source paragraphs or upstream QA strings. The preserved active corpus and all three
benchmark files match the import report's recorded input hashes.

## Signed-off official-source smoke — completed

Private notebook `annguyencherry/binancedoctalk-official-smoke-82d67381` version 1 is COMPLETE.
Private input dataset visibility was confirmed; only source/smoke/signoff inputs were attached.
Submitted source archive SHA-256:
`82d673816e932416d1365c61daed8d7bfe5c8a7956fbc4daf9d371e80558e712`.

Actual execution: 97 tests pass; 20 sources / 477 paragraphs form 229 BGE chunks; 10 smoke
questions produce 20 baseline/RAG predictions with zero structural errors. All raw output JSON
objects match their validated counterparts. Input, source, owner-decision and prediction hashes
were checked after download. Model outputs are ungraded; observed refusal classification
mismatches are retained in error analysis. The real BGE tokenizer/embedding path exercised the
protected definition/exception groups on these source snapshots successfully.

Runtime is Tesla T4, FP16, torch 2.11.0+cu128 / CUDA 12.8. Setup 114.48 s; illustrative query times
2.03–23.64 s; peak PyTorch allocated VRAM 8,980,763,136 bytes. API spend zero; GPU compute unpriced.
Full metadata is `kaggle_official_smoke_v2.json`; all predictions and the unfilled grading sheet
remain private. No official held-out consumption marker was created.
