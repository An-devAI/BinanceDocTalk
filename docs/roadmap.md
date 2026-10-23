# Seven-day implementation and release checklist

The code foundation is implemented. Remaining human-dependent work must still be done before
calling this a completed official-source portfolio study. Budget about 42 hours in total;
source review and annotation time vary, so remove UI polish/hosting before reducing review quality.

| Day | Owner outcome | Acceptance |
|---|---|---|
| 1 | Permitted source collection; API/model access; environment lock | Actual collection dates/hashes, reviewed scope, no copied credentials |
| 2 | Correct and bind 92 v2 drafts and audit evidence gaps; family/near-duplicate audit | Every label, reference and span inspected; 10/36/46 split; complete corpus audits; scope and injection separated |
| 3 | Trace live pipeline; inspect source/index; run smoke and tests | Working cited answers and explicit errors on a reviewed official corpus |
| 4 | Dev B0 vs RAG k=3/k=5; retrieval diagnostics; choose config | Human dev grades, recorded selection reason, no test tuning |
| 5 | Freeze inputs and run held-out once; grade 92 paired outputs | Original outputs retained, human grades, real metrics and denominators |
| 6 | Error analysis and clean-environment reproduction | Source version limits explicit; saved-output report reproduced |
| 7 | Review release, record final demo, prepare CV and publish | Public artifacts sanitized, licence split clear, honest commits |

## Before declaring the official-source portfolio study complete

- [ ] Confirm source and external API processing rights; no raw source redistribution without permission.
- [x] Lock the optional UI stack, pin local model revisions/packages, and record actual torch/CUDA versions.
- [x] Verify loopback HTTP startup and optional Streamlit AppTest; browser visual QA remains unverified.
- [x] Verify the chosen local Qwen/BGE backend with real Kaggle GPU inference; optional OpenAI remains mock-tested.
- [x] Record the owner chat signoff for 20 curated sources and 10 smoke references.
- [x] Complete private exploratory k=3/k=5 development inference: 144 retained outputs; semantic grades pending.
- [x] Implement and execute private AI-judge pilot; preserve gate rejection (0/12 full checks), with no semantic scores or fallback.
- [ ] Complete dev/test gold, gap audits and semantic near-duplicate inspection.
- [ ] Freeze before test; retain first run including errors.
- [ ] Complete human grading and publish sanitized measured artifacts.
- [ ] Update README results only from those artifacts.
- [ ] Add actual failure examples and limitations.
- [x] Run a signed-off official-source smoke on private Kaggle inputs and retain all 20 outputs.
- [ ] Record a reviewed official-corpus release demo; label any offline preview clearly.
- [ ] Inspect all files for secrets, account data and copied source text.
- [x] Initialize a real Git checkout on `main`; use an honest initial implementation commit.
- [ ] Publish a reviewed official-study release after these checks; hosting is optional.

The implementation is pushed on `main` to https://github.com/An-devAI/BinanceDocTalk.
This is an implementation preview; the official-source study and release checks above remain pending.
The initial commit records the existing implementation; subsequent commits should record real changes.

On 2026-10-22 the owner explicitly chose to leave human source/gold review pending while technical
integration is completed. The 20 cached source snapshots are preserved; curated v2 has 477 paragraphs and a
92-question review packet at `artifacts/local/review-pack-v3/`. Gap audits and gold remain pending. Do not run the held-out study or claim reviewed quality metrics until that review occurs.

The owner subsequently signed off the 20-source/10-smoke milestone. The exact statement and
bound input hashes are retained in `data/reviews/owner-smoke-v2.json`. Development/held-out
approvals and model-output grading are separate; none is inferred from that message.
