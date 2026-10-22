# Seven-day implementation and release checklist

The code foundation is implemented. Remaining human-dependent work must still be done before
calling this a completed official-source portfolio study. Budget about 42 hours in total;
source review and annotation time vary, so remove UI polish/hosting before reducing review quality.

| Day | Owner outcome | Acceptance |
|---|---|---|
| 1 | Permitted source collection; API/model access; environment lock | Actual collection dates/hashes, reviewed scope, no copied credentials |
| 2 | Correct and bind 80 gold drafts; family/near-duplicate audit | Every label, reference and span inspected; 8/32/40 split preserved |
| 3 | Trace live pipeline; inspect source/index; run smoke and tests | Working cited answers and explicit errors on a reviewed official corpus |
| 4 | Dev B0 vs RAG k=3/k=5; retrieval diagnostics; choose config | Human dev grades, recorded selection reason, no test tuning |
| 5 | Freeze inputs and run held-out once; grade 80 paired outputs | Original outputs retained, human grades, real metrics and denominators |
| 6 | Error analysis and clean-environment reproduction | Source version limits explicit; saved-output report reproduced |
| 7 | Review release, record final demo, prepare CV and publish | Public artifacts sanitized, licence split clear, honest commits |

## Before declaring the official-source portfolio study complete

- [ ] Confirm source and external API processing rights; no raw source redistribution without permission.
- [x] Lock the optional UI stack, pin local model revisions/packages, and record actual torch/CUDA versions.
- [x] Verify loopback HTTP startup and optional Streamlit AppTest; browser visual QA remains unverified.
- [x] Verify the chosen local Qwen/BGE backend with real Kaggle GPU inference; optional OpenAI remains mock-tested.
- [ ] Complete source/gold review and semantic near-duplicate inspection.
- [ ] Freeze before test; retain first run including errors.
- [ ] Complete human grading and publish sanitized measured artifacts.
- [ ] Update README results only from those artifacts.
- [ ] Add actual failure examples and limitations.
- [ ] Record an official-corpus demo; label any offline preview clearly.
- [ ] Inspect all files for secrets, account data and copied source text.
- [x] Initialize a real Git checkout on `main`; use an honest initial implementation commit.
- [ ] Publish a reviewed official-study release after these checks; hosting is optional.

The implementation is pushed on `main` to https://github.com/An-devAI/BinanceDocTalk.
This is an implementation preview; the official-source study and release checks above remain pending.
The initial commit records the existing implementation; subsequent commits should record real changes.

On 2026-10-22 the owner explicitly chose to leave human source/gold review pending while technical
integration is completed. The 20 cached source snapshots and 80-question review packets are ready
locally. Do not run the held-out study or claim reviewed quality metrics until that review occurs.
