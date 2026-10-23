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

## Curated v2 and the complete review packet

The parent snapshot and its original packet are historical artifacts. The active corpus is
`data/local/corpus.curated.v2.jsonl`; the active packet is `artifacts/local/review-pack-v3/`.
It contains all 20 sources/477 selected paragraphs, 92 drafts, per-document gap audits, policy,
feedback-selected evidence candidates, all split metadata and a semantic-overlap review shortlist.

Reproduce curation from the exact local parent without overwriting existing outputs:

```bash
python3 scripts/curate_corpus.py --input data/local/corpus.web-extract.v1.jsonl \
  --spec data/curation.v2.json --output data/local/corpus.curated.v2.jsonl \
  --manifest data/corpus.curated.v2.manifest.jsonl --log docs/corpus_curation_v2.json \
  --snapshots data/local/web-extracts
python3 scripts/prepare_benchmark_v2.py --corpus data/local/corpus.curated.v2.jsonl \
  --output data/eval/drafts-v2
python3 scripts/prepare_review.py --corpus data/local/corpus.curated.v2.jsonl \
  --output artifacts/local/review-pack-v3
```

These outputs already exist in this workspace; select fresh output paths for a reproduction.
The curation log retains removed/trimmed paragraph IDs, reasons and hashes without publishing source
text. The raw snapshot hash and literal Updated line provide date provenance. Grouped definition/
exception spans are retained as atomic source units when indexing; oversized groups fail explicitly.

Review all sources first, then whole-corpus gap audits, then the question references, labels,
required points and exact evidence. Use the commands in [evaluation.md](evaluation.md). Completed
partial decisions are saved on interruption; resume with the saved input and a new output version.
No previous draft review or owner feedback has been silently converted into gold approval.

## Continue the GPU study

Use `configs/kaggle-baseline.toml` and `configs/kaggle-rag.toml` for both arms. Build the reviewed
v2 dense index, run smoke, tune k=3/k=5 on dev, grade, then freeze the exact selected inputs.
When executing in Kaggle, provide reviewed data through authorized private inputs; do not embed
raw Binance documents into the public source bundle. Preserve the same paths, frozen source
fingerprint and held-out registry across sessions.

The runner binds consumption to question-set identity, not a freeze filename. Keep all results;
if held-out outputs influenced behavior, author a new independent final test. Official inference
and quality reporting remain gated on actual human source/gold/audit review and output grading.
