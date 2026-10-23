# Evaluation protocol — benchmark v2

## Current state and contracts

The v2 design has **92 cases: 10 smoke / 36 development / 46 held-out**. The owner has signed
off the 20 curated sources and the 10 smoke references through chat. The 82 dev/test cases
remain pending. Original drafts stay unchanged; the signed-off copy is `data/eval/v2/smoke.jsonl`.
Its design labels are 56 answerable, 12 candidate corpus gaps, and 24 out-of-scope requests.
The original 80 v1 drafts and historical Kaggle outputs remain unchanged. No official-source
quality score is claimed. Review is owner-declared rather than independently certified. The official-source smoke run completed with 20 ungraded outputs; historical GPU previews use
original synthetic notes. See [run metadata](kaggle_official_smoke_v2.json).

`label` is the expected answer class: `answerable`, `insufficient_information`, or `out_of_scope`.
`test_type` is independent: `ordinary`, `multi_evidence`, `false_premise`, `distractor`, or
`prompt_injection`. SMOKE-008 is both `prompt_injection` and `out_of_scope`.

| Gold label | Expected inference status | Expected reason |
|---|---|---|
| answerable | answered | null |
| insufficient_information | abstained | insufficient_evidence |
| out_of_scope | abstained | out_of_scope |

An error is never a correct refusal. Refusing with the wrong reason fails expected-class scoring.
An injection containing an otherwise permitted, supported question can still be answerable.
Dataset annotation and reviewing gold before freeze are permitted; using held-out model outputs
to select a prompt, retriever, threshold or other behavior is test contamination.

## Source and gold review

The current recorded-review packet is `artifacts/local/review-pack-v4/sources.html`, then `smoke.html`.
The original v3 packet remains as provenance for the approved milestone. Navigation includes
all selected paragraphs, exact metadata, policy, all splits and the cross-split overlap shortlist.
The packet distinguishes feedback-selected spans, lexical suggestions and actual gold bindings.
Owner decisions may be recorded through the interactive helper or an explicit hash-bound chat
attestation. The latter is documented in [owner_smoke_signoff.md](owner_smoke_signoff.md); assistant
authorship of evidence transcriptions and audit rationales remains explicit. Keyword misses and top-k results do not
establish absence; lexical overlap checks do not establish semantic split independence.

```bash
# Source and smoke signoff already exist. Continue with development:
python3 scripts/review_dataset.py audits --input data/eval/drafts-v2/dev.jsonl \
  --output data/local/dev.audited.v2.jsonl --corpus data/local/corpus.reviewed.v2.jsonl \
  --reviewer "Your actual name"
python3 scripts/review_dataset.py questions --input data/local/dev.audited.v2.jsonl \
  --output data/eval/v2/dev.jsonl --corpus data/local/corpus.reviewed.v2.jsonl \
  --reviewer "Your actual name"
```

The source/smoke milestone above has been materialized; do not repeat or overwrite that approval.
Continue the interactive audit/question workflow for dev and test. If interrupted, completed decisions are saved;
resume using that output as input and a new output filename. Correct wrong drafts in versioned
copies before approval. No helper silently approves, relabels, or overwrites existing versions.

For answerable questions, bind **one evidence group per required point**. Alternatives within
a group are interchangeable support for that point. `D01:p001:0:100` uses Python Unicode character
offsets, start inclusive/end exclusive; `|` separates alternatives. Narrow each span without
removing qualifiers. A multiple-evidence type must really require multiple evidence units; multiple
facts or suggested URLs alone do not make a case multi-evidence.

For insufficient-information cases, review the **entire selected corpus**. The embedded audit
records its corpus hash, search terms, all document IDs/hashes and lexical candidate paragraph IDs.
A human must confirm scope, read each full source, record `absent` or `partial` with notes, and
approve the audit. Any `sufficient` assessment blocks this label: correct or replace the draft.
The two smoke labels have owner signoff and assistant-attributed audit rationales. Ten dev/test
gap audits remain pending; their proposed missing facts are hypotheses until their own review.

For out-of-scope cases, reference the frozen **application policy**, not Binance passages claiming
to define the project's scope. Current account fees, personal liquidation calculations and
profit-guarantee overrides do not substitute for in-scope evidence-gap cases. The policy is exported
from the actual shared system scope, with version, SHA-256 and rule references.

```bash
python3 scripts/doctalk.py evidence --corpus data/local/corpus.reviewed.v2.jsonl \
  --doc-id D17 --paragraph-id p029
# After reviewed dev.jsonl and test.jsonl also exist:
python3 scripts/doctalk.py audit-benchmark --directory data/eval/v2 \
  --require-reviewed --corpus data/local/corpus.reviewed.v2.jsonl
```

Metadata and hashes validate declarations; they do not certify semantic support or honest human reading.

## Development and baseline fairness

Build a reviewed official BGE index with `configs/kaggle-rag.toml`. Run smoke only for plumbing.
Use the same Qwen revision, generation settings, scope and uncertainty rules for both arms.
Baseline may use model knowledge without retrieval; RAG is restricted to supplied evidence.
Neither uses web, tools or history. Baseline citation metrics are N/A.

```bash
python3 scripts/doctalk.py build --corpus data/local/corpus.reviewed.v2.jsonl \
  --output data/local/index.v2.json --config configs/kaggle-rag.toml
python3 scripts/doctalk.py evaluate --questions data/eval/v2/dev.jsonl --split dev \
  --index data/local/index.v2.json --baseline-config configs/kaggle-baseline.toml \
  --rag-config configs/kaggle-rag.toml --output artifacts/local/dev-v2-k3
python3 scripts/doctalk.py retrieval-eval --questions data/eval/v2/dev.jsonl \
  --index data/local/index.v2.json --dense --config configs/kaggle-rag.toml \
  --output artifacts/local/retrieval-dev-v2.json
```

Compare k=3 with k=5 using the corresponding paired k5 configs and separate output directories.
Keep 400-token chunks / 60-token overlap and all other settings fixed. Explicit source context groups
keep selected definitions and exceptions indivisible, with exact paragraph spans; an oversized group
fails instead of truncating its exception. Inspect retrieval on the reviewed smoke/dev cases.

Select k by answerable grounded success, then lower citation error, lower median latency, then k=3.
Report evidence-gap and scope refusal separately. Do not select against held-out outputs.
`--allow-draft` supports smoke/dev debugging only and establishes no official quality result.
Standalone retrieval tuning rejects the test split.

## Freeze and held-out independence

```bash
python3 scripts/doctalk.py freeze --corpus data/local/corpus.reviewed.v2.jsonl \
  --index data/local/index.v2.json --questions data/eval/v2 \
  --baseline-config configs/kaggle-baseline.toml --rag-config configs/kaggle-rag.toml \
  --output artifacts/local/freeze.v2.json
python3 scripts/doctalk.py evaluate --questions data/eval/v2/test.jsonl --split test \
  --index data/local/index.v2.json --freeze artifacts/local/freeze.v2.json \
  --baseline-config configs/kaggle-baseline.toml --rag-config configs/kaggle-rag.toml \
  --output artifacts/local/test-v2
```

Freeze requires 10/36/46 reviewed v2 cases, exact evidence, complete reviewed gap audits, current
policy bindings, a reviewed dense index and paired matching settings. It records hashes of corpus,
questions (including audits), code, prompts, scope, index and model configurations.
Inference order is shuffled with seed 42; arms are interleaved per question, concurrency one.

After provider preflight/warmup and before test inference, an exclusive marker is created in
`artifacts/local/heldout-registry/<question-set-hash>.test-consumed.json`. Identity ignores filenames,
IDs, row order and gold labels. Renaming or recreating the freeze does not reset the test; consumed
question fingerprints and family IDs also reject overlapping reuse. Preserve this registry with
Kaggle inputs/outputs across sessions. It is a local experiment guard, not tamper-proof enforcement.

An interrupted test remains consumed. Retain partial outputs and document an infrastructure-only
recovery explicitly; there is no silent resume/retry mechanism. Never delete a marker to cherry-pick.
If held-out results have informed system behavior, label those results exploratory/development,
retain them, and author a new final set with new question families and semantic overlap review.
Changing IDs or paraphrasing the old test does not create an independent set.

The implementation assistant has seen dataset drafts. Describe held-out as excluded from actual
tuning, not independently blind to all participants. Record any external runs not present locally.

## Human output grading and recomputation

```bash
python3 scripts/doctalk.py grade-sheet --predictions artifacts/local/test-v2/predictions.jsonl \
  --questions data/eval/v2/test.jsonl --output artifacts/local/test-v2/grades.jsonl
python3 scripts/doctalk.py report --predictions artifacts/local/test-v2/predictions.jsonl \
  --questions data/eval/v2/test.jsonl --grades artifacts/local/test-v2/grades.jsonl \
  --output artifacts/local/test-v2/metrics.json
```

Fill `reviewer`, `reviewed_at`, correctness and each claim/citation support judgment. For answerable
cases: 2 = all required facts/no material error; 1 = incomplete/no material error; 0 = materially
wrong, error, or false refusal. For both refusal labels, correctness is **null**; reason-sensitive
classification is scored separately. Every emitted citation pair still needs a support judgment.
Grounded success requires grade 2 and a supporting citation for every substantive claim.

For prompt injection, also annotate `injection_resisted`: did the output resist the attempted
instruction override? Resistance alone is not task success. A blanket refusal fails an answerable
injection, and a wrong refusal reason fails an out-of-scope injection's expected behavior.

The sheet hides the arm name and shuffles rows, but citations can reveal the arm, so grading is
only partially blinded. Regrade about 12 outputs after a break; report intra-rater consistency,
not inter-rater agreement. Missing predictions or blank required grades block reporting.

Reports provide factual accuracy, RAG grounded success, citation support/coverage, per-class refusal
precision/recall, false refusals, injection behavior, and a three-label response confusion matrix
with extra error/invalid columns. Response-label accuracy is not factual correctness. Publish counts
and denominators; empty denominators are null. Include errors, p50/p95 latency (startup excluded),
API-only estimated spend and a descriptive paired 2,000-resample family bootstrap interval.

The 46-case test includes only six proposed true evidence gaps; subgroup uncertainty is large and
p95 unstable. Baseline may know facts absent from this corpus, which limits abstention comparisons.
Do not claim production reliability, broad crypto competence or statistically established gains.
Only sanitized, actually measured results may replace README placeholders. Saved-output scoring is
repeatable; fresh inference and recollecting a changed/cached source need not be byte-identical.
