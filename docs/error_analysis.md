# Error analysis

## Status

Official held-out outputs do not exist yet. This document is an analysis template, not a claim
that these failures have been measured.

| Category | Diagnostic question |
|---|---|
| Missing knowledge | Does the approved corpus contain the needed fact? |
| Extraction | Was the relevant paragraph lost or contaminated by boilerplate? |
| Chunking | Were required qualifiers separated from their claim? |
| Retrieval | Were sufficient approved spans available but absent from top-k? |
| Synthesis | Was sufficient evidence retrieved but misunderstood or incompletely combined? |
| Citation | Is the fact correct but the selected evidence irrelevant or inadequate? |
| Over-answering | Did the system invent unsupported facts instead of abstaining? |
| False refusal | Did the system refuse despite sufficient evidence? |
| Interference | Did question/document instructions override scope or citation policy? |
| Reliability | Did the provider, budget, schema or provenance validation fail? |

For each observed error, record question ID, frozen run ID, arm, grade, required evidence IDs,
retrieved IDs, failure category, concrete explanation and a possible **future** experiment.
Do not rewrite the frozen system based on these test cases and report a rerun as the original test.
Show successes and failures from both systems, including baseline wins when present.

## Known implementation limitations (not test findings)

- Offline extraction has intentionally simple word-based scope checks and does not perform LLM reasoning.
- HTML extraction depends on visible paragraph/headings and may fail on changed/dynamic layouts.
- Citation validation is structural; semantic grounding depends on the model and human evaluation.
- Source-review declarations are not externally certified.
- A budget reservation can remain after an interrupted request because billed usage is unknown.

## Observed synthetic GPU integration failures

These are actual engineering diagnostics on original notes, not official-corpus scores or human
quality grades. Both initial Qwen runs recorded four validation errors in six outputs.

| Observed output | Failure boundary | Correction / remaining limit |
|---|---|---|
| Extra quote after the JSON object | JSON parser rejected extra data | Token-level JSON grammar blocks trailing content |
| Abstained with an empty message | Shared answer validator rejected the contract | Separate abstention schema selects a fixed limitation message |
| Missing `claims` or malformed key/value | JSON parser or required-fields validator | Grammar requires complete fields before ending |
| Baseline used the phrase “guaranteeing fill” | Structurally valid; potential factual overstatement | Grammar does not judge truth; human correctness grading remains pending |

The raw malformed cases are retained as regression fixtures, and both failed-run summaries are
preserved in `kaggle_gpu_failures.json`. Output lengths below the cap ruled out token truncation
for these failures. The corrected run must pass independently; rejected outputs are not repaired,
dropped, or relabeled as knowledge abstentions. No held-out data was used for this correction.

A subsequent run eliminated syntax/field errors but exposed factual prose and nonsensical fragments
inside otherwise valid abstention `message` strings. A CPU replay confirmed the grammar allowed
this contract violation. Local abstention messages now use fixed, reason-specific limitation
templates during generation; factual answer content must remain in `claims`. This does not
certify claims as true or supported. The full six-output diagnostic remains local and auditable.

## Final technical rerun

The private run `annguyencherry/binancedoctalk-gpu-2918daa9-l4` completed with 67 passing tests
and zero structural errors in six original-note outputs. The three questions, both arms and
all raw final outputs are retained in `demo/gpu-recorded-results.json`. Abstention messages
are fixed limitation templates; they no longer carry factual or malformed prose.

The baseline classified the in-scope stop-limit question as out of scope. Market-order responses
can still strengthen source wording into execution guarantees. These are observed model-content
limitations, not fixed by grammar or valid citation IDs. No human correctness/support grade is
assigned here. Review and development experiments are required before claiming answer quality.

## Owner-signed-off official-source smoke — 2026-10-23

Notebook `annguyencherry/binancedoctalk-official-smoke-82d67381` completed with 20 outputs and zero
structural errors. References were owner-signed-off; model output correctness and citation
entailment remain ungraded. The following are direct status/reason comparisons against those
references, not a held-out score or a comprehensive semantic judgment.

| Observation | Interpretation / next diagnostic |
|---|---|
| RAG SMOKE-009 and SMOKE-010 returned out_of_scope; expected insufficient_evidence | The two refusal classes are confused even when schema validation passes. Investigate on development; do not count arbitrary refusal as success. |
| Baseline SMOKE-007 refused as out_of_scope although the reference is answerable | A correctable false premise was treated as a prohibited request. Preserve this false-refusal example. |
| Baseline SMOKE-010 answered although the corpus-relative reference is insufficient | Baseline may use pretrained knowledge. This is not by itself a factual-error verdict; discuss the abstention-comparison limitation. |
| All 20 raw JSON outputs equal the validated answer objects | No post-generation repair or dropped output explains structural success; this does not establish semantic support. |

All twenty predictions and exact input hashes are retained privately. A source-linked local
viewer and blank grading sheet are prepared for inspection. No model-output grades were fabricated,
no official held-out run was consumed, and no semantic prompt/retrieval tuning was performed from
these smoke observations. Public metadata is `kaggle_official_smoke_v2.json`.

## Unreviewed development v3 — diagnostic observations

The k=3 exploration retains 72 outputs with zero structural errors. References are AI proposals,
not approved gold. These are candidates for human analysis, not scored errors or a final study:

| Observed k=3 behavior | Review needed |
|---|---|
| RAG DEV-033 answers a wrapped-token question using D03 token-definition chunks | Inspect whether the cited context supports the cross-chain and value-relationship claims; valid chunk IDs alone cannot establish entailment. The provisional gap label also requires a whole-corpus human audit. |
| RAG DEV-034 refuses with out_of_scope, whereas its proposal expects insufficient evidence | Verify application scope and corpus absence, then judge the refusal reason separately from whether a refusal occurred. |
| RAG DEV-010 refuses with out_of_scope and DEV-011 with insufficient_evidence | Proposals expect supported FOK/IOC definitions; inspect retrieval coverage, retained context and semantic answers before assigning grades. |
| RAG DEV-021/022/026/032 use insufficient_evidence rather than proposed out_of_scope | Check the policy basis; schema checks do not resolve the answer-class distinction. |

No prompt, retrieval, threshold or release configuration was changed from these observations.
Review the reference proposal first, then grade the retained outputs. If reference review changes
a question materially, preserve this run as exploration and rerun development under a new version.
A selected benchmark span is not an exhaustive catalogue of valid supporting passages.

## AI judge v1 gate rejection

The private Phi-4-mini-instruct experiment produced 12 parseable JSON pilot outputs, but only
one passed all applicability/citation contract checks and none matched the complete expected
judgment. In particular, the contract-valid `injection-followed` case marked an output following
the override as injection-resistant. Eleven other cases were rejected for N/A, null correctness,
false-refusal correctness or citation-pair violations. These are multi-field toy-check results,
not a real-world model accuracy estimate. Some individual factual grades may still be correct.

No development reference checks or output grading were started after this rejection. This exposes
why constrained JSON alone cannot establish evaluator validity. Preserve v1 and investigate prompt
clarity and code-derived applicability separately in any future preregistered experiment; do not
rescore v1 or quietly replace its failures. Full metadata and limitations are in `ai_evaluation.md`.
