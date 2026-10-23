# AI-only saved-output evaluation

The owner chose private Kaggle inference without a paid API while human reference/output review
is unavailable. This does not approve development gold or assign human grades. The human report
and dashboard reject records explicitly authored by an LLM. No held-out inference is performed.

## Preregistered judge and gate

The judge is `microsoft/Phi-4-mini-instruct`, revision
`cfbefacb99257ffa30c83adab238a50856ac3083`, rather than the Qwen answer model. It uses the existing
Transformers 4.57.6 runtime, native Phi3 implementation, no remote code, FP16/SDPA on CUDA:0,
greedy decoding, seed 42, 6,144 input tokens / 1,024 output tokens, constrained JSON. No model
training, quantization, CPU fallback, paid API or automatic JSON repair is enabled. GPU wall budget
is 10,800 seconds including load; checks occur between tasks, so an in-flight task may finish later.
The underlying GPU compute is unpriced, not assumed zero cost.

The 12 original toy pilot cases and four critical cases are frozen in source before execution.
All 12 must return valid contract-compliant JSON, at least 10 must match expectations and all four
critical cases must pass: an explicitly false fact, a valid citation ID with unrelated content,
a wrong refusal reason and a request manipulating the evaluator. These are assistant-authored
sanity checks, NOT human calibration or evidence of real evaluation accuracy. Pilot failure stops
before reference/output grading; retain diagnostics and keep all semantic results pending. Do not
tune this frozen run's prompt against development model answers to pass the gate.

## Reference-first flow and output judgments

The private bundle contains the signed-off 20-source snapshot, 36 unapproved development proposals,
the proposal provenance and two original 72-output development trials with run/summary manifests.
File hashes, complete pairs, source spans and raw responses are checked; no held-out inputs are
included in that bundle. Source text and raw model/judge outputs stay private.

Before exposing model outputs to the judge, check each reference against full supporting paragraphs
and protected definition/exception groups plus application scope. Two evidence-gap proposals are
checked against windows covering every paragraph in every source, including protected exceptions.
Partial relevant windows receive a joint assessment; overflow/error yields unknown, not absence.
Audit labels remain machine hypotheses, never proof of whole-corpus absence. Existing OCO wording
and ETH/USDT transfer flags remain unresolved even if the model agrees. Save/fingerprint eligibility
before grading; only reference-consistent cases enter conditional summaries. No question or reference
is silently rewritten after inspecting predictions.

Grade each candidate independently, hiding arm/trial identifiers. Correctness is 0/1/2/unknown
for answerable references and null for refusal references. Citation entailment is yes/no/unknown
for every actual claim/citation pair using precisely reconstructed retrieved context. Refusal reason
and injection resistance are independent judgments. Baseline has no citation requirement. A fully
valid chunk ID does not establish support. All records retain `grader_type=llm`; human reviewer
fields are absent. Complete, identical inputs share cached judgments; reuse is disclosed rather
than counted as independent assessments.

Twelve output records are selected with seed 42, three per arm/trial, for evidence-block order
perturbations. Count genuinely changed inputs and compare substantive labels separately from
explanatory wording. No-op reversals/cache reuse are disclosed; this is sensitivity, not accuracy.
The first pass remains the reported pass; no choice of the more favorable regrade is permitted.

## Commands and artifacts

```bash
python scripts/prepare_ai_judge.py --owner YOUR_KAGGLE_USERNAME \
  --development-inputs artifacts/local/kaggle/dev-inputs-v3 \
  --runs artifacts/local/kaggle/dev-output-v3 \
  --input-folder artifacts/local/kaggle/ai-judge-inputs-new \
  --notebook-folder artifacts/local/kaggle/ai-judge-notebook-new
```

Upload as a private dataset; verify remote privacy and exact dataset attachment before pushing
the private notebook. It invokes `scripts/judge_development.py` with the exact input-manifest hash.
A COMPLETE notebook may legitimately have `pilot_failed`, `budget_stopped` or infrastructure failure;
notebook completion alone is not a successful judge experiment. No paid fallback is allowed.
Existing directories are never overwritten. Judgments are recorded incrementally with raw JSON,
error, task hash, model revision and latency. Task identity includes prompt, schema, payload and
configuration; no fuzzy cache matching is used.

```bash
python scripts/ai_report.py --run artifacts/local/kaggle/ai-judge-output/ai-judge-run \
  --output docs/kaggle_ai_judge_v1.json \
  --html artifacts/local/ai-judge-review.html
```

The report checks artifact hashes and distinguishes `ai_judged_unvalidated` from `human_graded`.
Conditional summaries expose total/eligible/resolved denominators, unresolved references, unknown
judgments and errors. Pilot failure produces no semantic metrics. Public metadata has no raw
sources, answers or judge explanations; the optional HTML is local/private. A human review queue
retains all 36 references and 144 outputs, prioritizing existing flags and whole-corpus gap audits.
Future sample review can assess judge behavior; it cannot promote every uninspected case to gold.
No release configuration is selected. Human grading, reference approval and frozen held-out study
remain separate future work.

Model documentation: https://huggingface.co/microsoft/Phi-4-mini-instruct
Judge limitations: https://arxiv.org/abs/2306.05685
