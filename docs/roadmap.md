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

## Before GitHub publication

- [ ] Confirm source and external API processing rights; no raw source redistribution without permission.
- [ ] Install and lock optional dependencies; record environment versions.
- [ ] Verify optional Streamlit and the loopback UI on the owner's machine.
- [ ] Check the live API integration and fixed snapshot with the owner's key.
- [ ] Complete source/gold review and semantic near-duplicate inspection.
- [ ] Freeze before test; retain first run including errors.
- [ ] Complete human grading and publish sanitized measured artifacts.
- [ ] Update README results only from those artifacts.
- [ ] Add actual failure examples and limitations.
- [ ] Record an official-corpus demo; label any offline preview clearly.
- [ ] Inspect all files for secrets, account data and copied source text.
- [x] Initialize a real Git checkout on `main`; use an honest initial implementation commit.
- [ ] Publish the GitHub repository only after these checks; hosting is optional.

A local Git repository is initialized on `main`. No GitHub remote has been configured or pushed.
The initial commit records the existing implementation; subsequent commits should record real changes.
