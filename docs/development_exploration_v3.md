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
