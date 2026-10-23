# Development proposal v3 and exploratory GPU runs

The v2 development drafts remain unchanged. `data/eval/proposals/dev.v3.jsonl` is an
assistant-authored, unapproved annotation proposal: 36 development cases, 25 answerable,
nine out of scope and two provisional evidence gaps. It is not gold. Every answerable
required point has one source-bound evidence group with character offsets and content hash.
The source corpus remains the 20 owner-signed-off snapshots; HF candidates are separate.

The change manifest records old/new questions, labels and references plus parent/source hashes.
DEV-035/036 were rewritten to source-supported questions because confirmed-record changes and
malicious validators losing stake are present in the corpus. Absence of the exact keywords
“finality” or “slashing” was not a valid absence-of-knowledge argument. DEV-016/031 flag
conflicting OCO cancellation wording in D18:p002/p009 for human adjudication. DEV-008 flags
application of the illustrated BTC/USDT convention to ETH/USDT. The two remaining provisional
gaps have a census of all 20 sources with AI-authored assessments; these are not human audits.

The private review packet is `artifacts/local/review-pack-dev-v3b/dev.html`; it links the full
corpus, policy, change manifest and other splits. The earlier owner approval covers sources
and smoke only. Do not promote this proposal, its gap audits or pending model grades to gold.

## Reproduce the preparation

Use the locally retained reviewed corpus. No source text is distributed in the code archive.

```bash
python scripts/prepare_development_experiment.py \
  --owner YOUR_KAGGLE_USERNAME \
  --corpus data/local/corpus.reviewed.v2.jsonl \
  --proposal data/eval/proposals/dev.v3.jsonl \
  --changes data/eval/proposals/dev.v3.changes.json \
  --input-folder artifacts/local/kaggle/dev-inputs-new \
  --notebook-folder artifacts/local/kaggle/dev-trials-new
```

Upload the input bundle as a **private** Kaggle dataset using an account with permission to
process these snapshots. Confirm its remote privacy before pushing either generated notebook.
The notebooks are private and each runs 72 paired outputs (36 baseline, 36 RAG). k=3 and k=5
share the pinned Qwen/BGE models and settings; their separate workers make latency comparisons
descriptive. Existing outputs cannot be overwritten. No paid API calls are enabled.

Each run retains raw predictions, exact configs/runtime information, provisional retrieval
coverage and a blank human grade sheet. Retrieval diagnostics depend on the unapproved spans;
valid JSON and citation IDs do not establish factual correctness or citation entailment.
`official_quality_measured`, `configuration_selected` and `heldout_inference_performed` remain
false. These runs do not select a release configuration. Human reference review and paired
output grading must precede selection; freeze and run the untouched held-out only afterward.

Use `scripts/prepare_prediction_review.py` with the corpus, proposal, each predictions file and
a new output HTML path to inspect exact reference excerpts and all retrieved contexts locally.
It preserves review status, includes errors, requires complete pairs and never assigns grades.

After download, `scripts/summarize_development.py --run RUN_FOLDER --questions
data/eval/proposals/dev.v3.jsonl --output NEW_METADATA.json` validates pairs, raw JSON, citation IDs,
proposal hashes and unreviewed flags before emitting allowlisted metadata. It does not calculate
answer accuracy or semantic citation support. Run files retain blank grade sheets privately.

## Recorded execution

Both private notebooks use source archive SHA-256
`1c83bab0ca33697966c63786728a4a8ee45d929bb6584a2fee47a107ba55eec2`
from commit `e5171ec`. The submitted code passes 102 tests. Input manifest SHA-256 is
`6e92999ec2a22b9eb149b22b5c4d9b10c56d4a77a1ead56e4fa3874d76b993de`.
Private dataset `annguyencherry/binancedoctalk-dev-inputs-61f62e0300f7`, version 1, was
confirmed ready and private through the server API. Both notebook metadata records were pulled
back and confirmed private with only that dataset attached. Underlying documents are not released.

The k=3 notebook is `annguyencherry/binancedoctalk-dev-k3-1c83bab0`, version 1, COMPLETE.
Its 72 predictions pass raw/validated JSON and citation-ID checks with zero structural errors.
Exact hashes, model/package versions, configs, execution conditions, descriptive timings and
**unreviewed-reference** retrieval diagnostics are in [k=3 metadata](kaggle_development_k3_v3.json).
The local review viewer is `artifacts/local/kaggle/dev-output-v3/k3/development-run/review.html`.
Semantic grading, reference approval, configuration selection and held-out evaluation are pending.

The k=5 notebook is `annguyencherry/binancedoctalk-dev-k5-1c83bab0`, version 1, COMPLETE.
[k=5 metadata](kaggle_development_k5_v3.json) records its 72 structurally validated outputs and
zero structural errors. Both workers ran Tesla T4, FP16, torch 2.11.0+cu128 / CUDA 12.8;
API spend is zero, GPU compute is unpriced. All 36 baseline raw responses match between workers;
this observation does not promise cross-hardware determinism. The two retrieval diagnostic files
match (same index/reference proposal, each evaluates both k cutoffs). These remain provisional.

| Trial | Baseline outputs | RAG outputs | Structural errors | Baseline p50 / p95 (s) | RAG p50 / p95 (s) |
|---|---:|---:|---:|---|---|
| k=3 | 36 | 36 | 0 | 4.24 / 12.71 | 15.37 / 30.61 |
| k=5 | 36 | 36 | 0 | 4.08 / 12.15 | 17.08 / 29.85 |

Timings are descriptive: one interleaved sequential pass, separate workers, concurrency one,
warm models, no response cache. p50 is the median; p95 uses nearest rank. Setup and model load
are separate. Output lengths/mixes differ, so these are not controlled throughput or speed gains.
Neither answered/abstained counts nor structural validity establish factual correctness.

Start local inspection at `artifacts/local/development-v3/index.html`. It links the reference
packet first, then each private source-linked output viewer and unfilled grade sheet.
All 144 original outputs, raw JSON, run manifests and logs are retained under
`artifacts/local/kaggle/dev-output-v3/`; no held-out consumption marker was created.
