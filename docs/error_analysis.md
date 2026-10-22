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
