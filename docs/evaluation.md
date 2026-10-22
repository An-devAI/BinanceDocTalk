# Evaluation protocol and operator guide

## Current state

No official inference, human gold review or human output grading has been completed. Test fixtures
exercise the software only. No test-derived accuracy, citation, latency or cost estimate is claimed.

## Review the data before running experiments

After reviewing the official corpus, review each split separately. This helper displays draft
references and candidate paragraphs; you must check them yourself. Correct a wrong draft in the
input JSONL before approving it. Work on copies to preserve original drafts.

```bash
python3 scripts/review_dataset.py questions --input data/eval/drafts/smoke.jsonl \
  --output data/eval/v1/smoke.jsonl --corpus data/local/corpus.reviewed.v1.jsonl \
  --reviewer "Your name"
```

Repeat with dev and test. `evidence_groups` require exact local spans; `D01:p001:0:100` denotes
one span, and `|` separates alternative spans for the same required fact. Separate input lines
create separate required evidence groups. Inspect every selected excerpt before approving.

```bash
python3 scripts/doctalk.py evidence --corpus data/local/corpus.reviewed.v1.jsonl \
  --doc-id D17 --paragraph-id p003
python3 scripts/doctalk.py audit-benchmark --directory data/eval/v1 \
  --require-reviewed --corpus data/local/corpus.reviewed.v1.jsonl
```

The tool validates declarations and references, not whether a human truly read them or whether
they entail the answer. A name/timestamp is an audit record, not certification.

## Development

Run smoke only for plumbing. Run both generation arms on development and compare RAG k=3 versus
k=5; keep chunk size 400 and overlap 60 model-tokenizer tokens. Do not change multiple factors and
attribute the result to retrieval alone. Set top_k consistently in both configs so the freeze
captures identical non-system settings.

```bash
python3 scripts/doctalk.py evaluate --questions data/eval/v1/dev.jsonl \
  --split dev --index data/local/index.v1.json --output artifacts/local/dev-k3
python3 scripts/doctalk.py retrieval-eval --questions data/eval/v1/dev.jsonl \
  --index data/local/index.v1.json --dense --output artifacts/local/retrieval-dev.json
```

Use a separate output directory for k=5. `--allow-draft` exists for smoke/dev debugging, but such
runs are unreviewed and cannot establish an official quality result. Standalone retrieval tuning
rejects the test split. Final test retrieval recall can be computed from saved RAG hits without
performing additional inference or using it for tuning.

Grade development outputs with the same rubric as test. Select the RAG k maximizing the mean of
answerable grounded success and insufficient-information abstention recall; ties select k=3.
If both fail important demonstration scenarios, fix the system on dev before freezing.

## Baseline fairness

Both arms use the same model, reasoning setting, maximum output, educational scope, and uncertainty
instructions. Baseline is explicitly allowed to use model knowledge; it is not constrained to an
empty evidence context. RAG is evidence-restricted. Neither uses web/tools/history.

Baseline citation metrics are N/A. Compare factual correctness and abstention separately from
RAG-only grounded success. Insufficiency is relative to this corpus/task: the baseline may know
a fact that is absent from the corpus, so discuss this limitation when comparing abstention.

## Freeze before opening test outputs

```bash
python3 scripts/doctalk.py freeze --corpus data/local/corpus.reviewed.v1.jsonl \
  --index data/local/index.v1.json --questions data/eval/v1
python3 scripts/doctalk.py evaluate --questions data/eval/v1/test.jsonl --split test \
  --index data/local/index.v1.json --freeze artifacts/local/freeze.v1.json \
  --output artifacts/local/test-v1
```

Freeze requires 8/32/40 reviewed questions, bound evidence, a reviewed official dense index,
and comparable baseline/RAG configs. It records input file hashes, source/code fingerprints,
prompt hashes, model settings and Git commit when available. Inference order is shuffled with
seed 42; one request sequence runs at a time and arms are interleaved per question.

`freeze.v1.test-consumed.json` is created exclusively before test inference. Do not delete it to
rerun and cherry-pick results. An interrupted/failed test remains an auditable consumed test;
preserve partial outputs and explicitly document an infrastructure-only recovery, or create a
fresh benchmark for behavioral changes. This implementation deliberately has no silent test-resume switch.
Missing API credentials fail before consuming the marker.

Development runs do not require a freeze. After viewing test results, do not use them to tune v1.
The original question drafts were seen by the coding assistant; describe the split as held out
from actual tuning, not independently blind or previously unseen by all participants.

## Human grading and offline recomputation

```bash
python3 scripts/doctalk.py grade-sheet --predictions artifacts/local/test-v1/predictions.jsonl \
  --questions data/eval/v1/test.jsonl --output artifacts/local/test-v1/grades.jsonl
```

Edit grades: `correctness`, each `citation_support[].supports`, `reviewer`, `reviewed_at`, and
notes. Open exact local evidence from the index/corpus. The sheet omits the arm label and is shuffled,
but citation presence can reveal the arm; grading is therefore only partially blinded. Regrade about
12 outputs after a break. This is intra-rater checking, not inter-rater agreement.

For answerable questions: 2 = all required facts and no material error; 1 = incomplete but no
material error; 0 = wrong/materially misleading, erroneous output, or false refusal.
Errors/refusals on answerable questions must receive zero. Insufficient cases do not use this
factual-accuracy denominator; abstention is measured separately.

Each emitted claim/citation pair gets a Boolean support annotation. Support means the cited
text actually supports that claim, including qualifiers. A valid source ID alone is insufficient.
An answerable output succeeds as grounded only if correctness is 2 and every substantive claim
has at least one supporting citation.

```bash
python3 scripts/doctalk.py report --predictions artifacts/local/test-v1/predictions.jsonl \
  --questions data/eval/v1/test.jsonl --grades artifacts/local/test-v1/grades.jsonl \
  --output artifacts/local/test-v1/metrics.json
```

Reporting requires every prediction, grade and paired question. Missing outputs cannot be dropped.
Blank support labels or correctness grades block a report. AI can suggest annotations; final grades
must be inspected by the owner. Copy sanitized artifacts to `artifacts/results/<run-id>/` for release.

## Metrics

- Evidence recall@3/@5: fraction of required evidence groups covered; any approved alternative
  suffices. Union character spans across overlapping chunks without double counting.
- Complete evidence rate, with a separate multi-evidence subset.
- Full answer accuracy: count of grade-2 outputs / all answerable questions, including errors/refusals.
- RAG grounded success: correct-and-supported outputs / all answerable questions.
- Citation support precision: supported pairs / emitted pairs. Coverage: claims with support / claims.
  Empty denominators are null, not perfect scores; baseline citation metrics are null.
- Abstention precision/recall, false refusal rate on answerable questions, answer rate and error counts.
- p50/p95 end-to-end latency, including query embedding, retrieval, generation, validation and retries;
  indexing/startup excluded, document index preloaded, response cache disabled. Errors remain counted.
- Estimated cost from actual usage at recorded prices; failed unknown usage retains conservative reserves.
- Paired accuracy difference with a 2,000-resample family bootstrap, seed 42, descriptive 95% interval.

Always publish denominators, errors and per-category examples. With only 28 answerable and 12
insufficient test cases, uncertainty is wide; p95 is especially unstable. Do not claim general
crypto competence, production reliability, statistical significance, or investment usefulness.

Saved-output recomputation is deterministic. Fresh API inference is not guaranteed byte-identical,
even with a model snapshot. The model may already have encountered public sources during pretraining.
