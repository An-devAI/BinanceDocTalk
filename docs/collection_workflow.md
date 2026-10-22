# Source collection and human review handoff

## Observed collection attempt

All 20 candidate URLs were attempted on 2026-10-22. Zero usable articles were extracted and no
corpus was created. A representative direct page request returned HTTP 202 with an empty body.
`docs/source_collection_run.json` records the failures without source text. Robots policy allowed
the article paths at the time of the attempt; this is not a content redistribution license.
No challenge bypass, proxy rotation, or automatic review approval was used.

## Retrieved web-reader snapshots

The web reader subsequently returned complete indexed text for all 20 official sources.
These were imported into ignored `data/local/corpus.web-extract.v1.jsonl`: 481 paragraphs,
all pending human review. The raw indexed text is retained in `data/local/web-extracts/`.
The public `data/corpus.web-extract.v1.manifest.jsonl` contains only metadata and hashes.

These extracts may be cached. `fetched_at` records our snapshot retrieval time, not a claim
about the upstream crawler’s exact fetch time or the live page’s freshness. Each paragraph
retains its indexed snapshot line in local provenance. The importer rejects missing lines,
removes navigation/related-content blocks and procedural code, and preserves scope filters.

```bash
python3 scripts/import_web_extracts.py --folder data/local/web-extracts \
  --output data/local/corpus.web-extract.v2.jsonl
python3 scripts/prepare_review.py --corpus data/local/corpus.web-extract.v1.jsonl \
  --output artifacts/local/review-pack-v3
```

The already generated `artifacts/local/review-pack-v2/` includes the actual 20 snapshots and
80 drafts, with evidence candidates per required point. The packets do not approve any record.
Raw web-reader extracts cannot be reconstructed by an HTTP-only script when the website blocks
that collector. Recollect with an authorized web reader or use the HTML import route below;
new source content requires a new manifest/hash and evidence review.

## Import snapshots obtained through ordinary authorized access

Use real saved HTML, not an empty challenge page, PDF printout renamed as HTML, or an AI summary.
Keep files in ignored `data/local/snapshots/`, with matching names:

- `D01.html`: actual HTML for the D01 source.
- `D01.metadata.json`: actual final `url` and actual `fetched_at` (ISO 8601, timezone required).

A sidecar shape is `{"url":"<actual final Binance article URL>","fetched_at":"<actual collection timestamp>"}`.
The importer does not invent a collection time from the import time or a file modification date.

```bash
python3 scripts/import_sources.py --folder data/local/snapshots \
  --output data/local/corpus.imported.v1.jsonl
python3 scripts/prepare_review.py --corpus data/local/corpus.imported.v1.jsonl \
  --output artifacts/local/review-pack-v2
```

The read-only packet separates smoke/dev/test, shows draft references, and suggests exact paragraph
spans ranked with BM25. These are search suggestions, not semantic entailment judgments or gold.
Open `sources.html` first; inspect extraction, actual page correspondence, coverage and stable scope.
Out-of-scope labels require inspecting the entire selected corpus, not merely the top candidates.
If a proposed question is unsupported, correct or replace it before freezing the benchmark.

A zero-corpus packet has already been prepared locally under `artifacts/local/review-pack-v1/`.
It contains all 80 drafts and proposed source links, with an explicit missing-corpus warning.

## Record actual review decisions

```bash
python3 scripts/review_dataset.py sources --input data/local/corpus.web-extract.v1.jsonl \
  --output data/local/corpus.reviewed.v1.jsonl --reviewer 'Your actual name'
python3 scripts/review_dataset.py questions --input data/eval/drafts/smoke.jsonl \
  --output data/eval/v1/smoke.jsonl --corpus data/local/corpus.reviewed.v1.jsonl \
  --reviewer 'Your actual name'
```

Repeat question review for dev and test, retaining 8/32/40 cases and family isolation. The helper
only saves approvals entered by the operator. Corrections belong in versioned copies, not silently
rewritten historical artifacts. Afterward run the reviewed benchmark audit documented in
`docs/evaluation.md`. Do not use the coding assistant's review as a human review declaration.

## Continue the GPU study

Use `configs/kaggle-baseline.toml` and `configs/kaggle-rag.toml` for both arms. Build the dense index
from the reviewed corpus, run smoke, tune k=3/k=5 on dev, grade, then freeze the exact selected inputs.
When executing in Kaggle, provide the reviewed data through authorized private inputs; do not embed
raw Binance documents into the public source bundle. Record the actual input paths in that session.
The freeze and held-out run must use the same preserved paths and source fingerprint.

Run held-out once only after the source/gold audit passes. The freeze refuses unreviewed draft gold;
the report refuses ungraded predictions. Final metrics and a real official-source demonstration
remain pending until these independent review and inference steps are complete.
