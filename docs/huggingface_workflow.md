# Pinned Hugging Face source and development candidate import

This optional import stages external material for review. It **does not replace the active
20-source curated corpus, merge QA into benchmark v2, assign gold labels, or run a model**.
The active 92-case smoke/dev/test split and its evidence-gap audits remain unchanged.

## Inputs and actual observed selection

| Dataset | Pinned revision | Pinned Parquet rows |
|---|---|---:|
| [Crypto Education Corpus](https://huggingface.co/datasets/kskada/crypto-education-en-corpus) | `f23dd9dd360b485fb25db063ed0a3dad38d4dbd0` | 3,487; 581 Binance Academy |
| [Crypto Education Golden Set](https://huggingface.co/datasets/kskada/crypto-education-en-golden-set) | `5f22df94f486328c5baaf2f5bdb64b2cc5afa893` | 497; 84 QA across 42 Binance URLs |

`data/huggingface.sources.v1.json` pins revision, Parquet filename, byte count, SHA-256, row count,
columns and supplemental source allowlist. The standalone corpus `train.jsonl` had only 66 rows
at the inspected revision; the importer deliberately reads the full pinned Parquet instead.
The dependency is optional `pyarrow==26.0.0`; the core app does not depend on Arrow or the HF SDK.

The import selects **15 source candidates**: nine URL matches with the current manifest and six
explicit supplemental articles. These cover general order types, stablecoins, proof-of-stake,
Ponzi/pyramid terminology, spoofing and Ethereum basics. Inclusion is a shortlist, not approval
of every section: procedural, promotional, legal, numerical and investment content needs review.
The potential union is 26 URLs, before semantic deduplication and scope filtering; it is not an
active 26-document corpus.

There are **12 external QA candidates**, all proposed for development. Their sources are the six
supplemental articles. None of the 84 upstream Binance QA rows directly join the current 20 URLs
under the implemented hostname/path normalization. Different slugs or equivalent content can still
overlap semantically. Full local comparisons show the nine matched current/mirror snapshots;
URL equality and token Jaccard are not content correctness or freshness checks.

## Reproduce without overwriting tracked metadata

```bash
python3 -m venv .venv-data
.venv-data/bin/python -m pip install -r requirements-datasets.txt
.venv-data/bin/python scripts/import_huggingface.py \
  --output artifacts/local/hf-repro-v1 \
  --public-manifest artifacts/local/hf-repro-v1/public/sources.jsonl \
  --public-qa-manifest artifacts/local/hf-repro-v1/public/qa.jsonl \
  --public-report artifacts/local/hf-repro-v1/public/inspection.json
```

With the preserved local current corpus, add
`--current-corpus data/local/corpus.curated.v2.jsonl` for side-by-side text comparisons.
Without it, source/QA selection and matching still work; current-text equality/similarity stays null.
Use new output paths for each import. Existing outputs are rejected; failed drafts are preserved.
Verified download caches are reused without inventing a newer retrieval timestamp. A cache missing
its provenance sidecar, a changed file hash, or an unexpected schema/count fails closed.

The current prepared packet is `artifacts/local/hf-import-v1/review/index.html`:

- `sources.html`: all selected mirror paragraphs with provenance and review status.
- `comparisons.html`: current vs mirror text for the nine URL matches.
- `candidates.html`: original upstream questions/answers, review flags, source links and lexical
  overlap candidates from the current smoke/dev/test question inputs. No held-out predictions are read.

Actual imported text and upstream QA stay under ignored `artifacts/local/`. The public files
`data/huggingface.candidates.v1.manifest.jsonl`, `data/huggingface.qa-candidates.v1.manifest.jsonl`
and `docs/huggingface_import_v1.json` contain metadata, hashes, selection diagnostics and counts.
They contain neither full source text nor copied upstream question/answer strings.

## Provenance and interpretation

The `url` remains the official URL attributed by the upstream dataset; it is not rewritten into
an unobserved redirect. Legacy `academy.binance.com/en/articles/` HTTPS routes are allowlisted.
Metadata clearly identifies `collection_method=huggingface_mirror`, repo/revision/file/row index,
file/row/markdown hashes and Python-character spans into the original markdown row. The import
only segments literal markdown blocks and trims whitespace; it does not summarize missing text.

`retrieved_at` and `fetched_at` mean **our HF artifact download time**. Original page crawl time
and `source_updated_at` remain null. Dataset upload dates and file mtimes are not page timestamps.
No existing source review is inherited by a mirror with different content/provenance.

The publisher declares MIT on both dataset cards and describes the QA as Claude-generated from
sampled corpus documents. The card does not describe manual gold verification. We record that
license declaration; underlying third-party article rights have not been independently verified.
See [the separate notices](../NOTICE.md) and [dataset card](dataset_card.md).

## Review before any adoption

The external candidate format is intentionally not a benchmark row: `label`, `test_type` and
required facts are unassigned, final evidence is empty, and review status is pending. Upstream
`question_type` is linguistic style, not our behavioral test type or answerability label.
Evaluation rejects the external candidate format; retrieval rejects question records.

1. Check official-page correspondence, source scope, extraction quality and version differences.
   Compare stablecoin article/glossary overlap and other semantic duplicates before expanding sources.
2. Review and, when necessary, rewrite each QA. Template artifacts, unanswered comparisons,
   reused answers, legal assertions, absolute execution claims and maker/taker qualifications are
   flagged as review prompts, not automatic correctness judgments. Preserve the upstream record.
3. Decide project label/type, required facts and exact supporting spans. All related variants
   from a source are grouped together; manually clear semantic overlap with existing test families.
4. Any adopted sources require a new corpus/index version. Re-audit **all existing evidence gaps**
   because new text can make a formerly unanswerable question answerable; rebind affected evidence.
5. Any adopted QA requires an explicitly versioned benchmark and split-size update. Do not append
   it silently to the fixed 10/36/46 experiment. Use it for development; retain an independent final
   held-out set and the existing consumption registry. No held-out outputs may guide tuning.

## Validation and attribution

Core tests use original fixtures and mocked network responses to check pins, cache provenance,
URL safety, selection bounds, candidate isolation, Unicode spans and HTML escaping. CI also runs
`python scripts/check_huggingface_import.py` with the real Arrow reader on a tiny original Parquet
fixture; it does not redownload the upstream datasets. The actual pinned files were imported locally,
and their measured counts/hashes are retained in the sanitized inspection report. None of these
checks measures official-source LLM quality.

Credit Kirill Konovalov / `kskada` for the two Crypto Education datasets (2026). The corpus card
identifies `distilled-ai/web3-oriented-pretraining-data` as an upstream source for Binance articles.
Keep those dataset citations and provenance when deriving future dataset versions.
