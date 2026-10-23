# Owner signoff: source corpus and smoke milestone

The owner replied **“tôi duyệt. triển khai đi”** after the proposed next milestone of 20 curated
sources and 10 smoke cases. That actual statement is recorded, without extending it to development,
held-out, HF candidate adoption, output grading, or source redistribution permission.

- `data/reviews/owner-smoke-v2.json` records the exact statement, interpreted scope, policy hash,
  approved source/question IDs and hashes of the input corpus and smoke packet data.
- `data/local/corpus.reviewed.v2.jsonl` is the private signed-off corpus; its public metadata is
  `data/corpus.reviewed.v2.manifest.jsonl`. Original source text and version hashes are unchanged.
- `data/eval/v2/smoke.jsonl` records ten smoke references: five answerable, two insufficient-information,
  three out-of-scope. Existing feedback-selected spans were bound without changing question text,
  labels, references or required points. The two corpus-gap rationales enumerate all twenty sources.

This is **owner-declared approval, not independent certification**. The owner did not author each
new annotation: evidence transcription and per-document audit rationale authorship are explicitly
attributed to the assistant. No claim is made that the owner wrote or separately inspected each
new rationale. `reviewed_at` records the time the approval was entered into the dataset, not a
fabricated duration or timestamp for reading individual pages. The original pending drafts and
v3 review packet remain preserved; the v4 packet displays recorded approvals alongside pending data.

The helper `scripts/apply_smoke_signoff.py` refuses changed input/policy hashes, expanded scope,
missing candidate bindings, incomplete per-source rationale coverage and existing output versions.
It has no implicit owner statement or default reviewer approval. The ordinary interactive review
helper remains available for dev/test; explicit smoke signoff does not approve those cases.

## Private Kaggle smoke execution

`prepare_official_smoke.py` validates the exact approved inputs and creates a private-only input
bundle containing source corpus, ten smoke references, owner decision and an integrity manifest.
No dev/test inference inputs are uploaded in that dataset. The public code archive contains code,
metadata and original question/reference data; raw Binance articles are not embedded in notebook code.
The private dataset's visibility is checked remotely before notebook submission.

`official_smoke.py` requires 20 sources and 10 smoke cases, validates source/gold/audit hashes,
builds the real pinned BGE index, verifies chunk provenance, and runs paired baseline/RAG generation
with shared Qwen weights. It never freezes or runs held-out. Setup, actual runtime, all predictions,
errors and raw model responses are retained. An embedding failure cannot inherit an older query's
raw response. Model outputs remain **ungraded**; no final accuracy, citation-support score or RAG
improvement is inferred from this pipeline check.

The updated local review packet is `artifacts/local/review-pack-v4/smoke.html`. For the next phase,
review the 36 development and 46 held-out references separately, retain family independence, and
select behavior using development only. Review model outputs before reporting quality metrics.

## Observed run

Private notebook `annguyencherry/binancedoctalk-official-smoke-82d67381` version 1 completed.
Dataset and notebook privacy were confirmed remotely. Exact run metadata is published in
`kaggle_official_smoke_v2.json`; original predictions/raw responses and a blank human grading
sheet remain in ignored `artifacts/local/kaggle/official-smoke-output-v2/official-smoke-run/`.
The private `review.html` there displays all twenty outputs with cited snapshot excerpts.

Observed: 97 core tests pass, 229 real BGE chunks, 20 raw/validated outputs match, zero structural
errors. The run used Tesla T4 / FP16 / torch 2.11.0+cu128 / CUDA 12.8. Setup took 114.48 seconds;
individual illustrative requests took about 2.03–23.64 seconds, not a representative latency study.
API spend was zero; GPU compute remains unpriced. RAG used out_of_scope for both approved evidence-gap
cases, and baseline refused the answerable stop-limit false-premise case. These mismatches are
retained as smoke diagnostics; they are not a substitute for development or held-out grading.
No prompt, retrieval setting or refusal threshold was changed based on these observations.
