# Verification record — 2026-10-22

## Completed locally

| Check | Observed result |
|---|---|
| Core and integration suite | 59 tests pass on Python 3.12.3 |
| Benchmark structural audit | 80 drafts; splits 8/32/40; labels 56/24; zero declared human reviews |
| CLI demonstration | Actual original-note retrieval, false-premise example and out-of-scope response |
| API transport | Mocked structured generation, embedding order, usage, 401/no retry, 429/one retry |
| Held-out controls | Review/freeze gates, payload integrity, immutable first-run marker tested on synthetic fixtures |
| UI server handlers | In-memory handler tests plus actual loopback server startup: HTTP 200, 12,043-byte page |
| Browser JavaScript | Syntax checked with Node.js; not browser interaction testing |
| Python package | Wheel builds without fetching runtime dependencies; HTML asset included |
| Core lockfile | `uv lock --check --offline` passes; no runtime dependencies |
| Kaggle notebook | Private CPU run COMPLETE; 45 tests pass on Python 3.13.15; three demo cases executed |
| Demo video | 27-second VP8 WebM, 1280×720, inspected with GStreamer; original-note CLI walkthrough |

The test suite uses original/synthetic fixtures and mocked network responses. Its numerical
assertions validate the metric implementation; they are not project accuracy or citation scores.

## Not verified / not completed

- No actual OpenAI call: no OpenAI key is configured in the implementation environment.
- Official collection attempted 20 URLs: zero usable articles; see `docs/source_collection_run.json`.
  No official corpus or human approval is claimed.
- No human gold annotation, official held-out inference or semantic output grading is complete.
- Streamlit and tiktoken are declared optional requirements but were not installed here.
- No real browser UI inspection: browser control reported no available browser, and local browser
  process startup was blocked. The poster is a rendered CLI presentation, not a browser screenshot.
- The implementation was pushed to GitHub on `main`; remote Actions results and online hosting
  have not been verified.

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

## Reproduce local checks

```bash
python3 -m unittest discover -s tests -v
python3 scripts/doctalk.py audit-benchmark
python3 scripts/doctalk.py demo
python3 scripts/kaggle_task.py doctor --check-auth
```

Raw local check logs, budget/configuration state and remote outputs belong under ignored
`artifacts/local/`. Inspect any future public results against the frozen manifest and human grades.
