# Dataset card

## Three separate assets

| Asset | Location | Version/status |
|---|---|---|
| Candidate official-source manifest | `data/manifest.v1.jsonl` | 20 URLs, checked as candidates on 2026-10-22; not collected |
| Benchmark | `data/eval/drafts/*.jsonl` | 80 AI-authored drafts, no human-reviewed gold evidence |
| Original preview corpus | `data/demo/notes.jsonl` | 12 synthetic notes; not official Binance material |

No training/fine-tuning dataset is used. Never add question/reference-answer files to the index.
All official data is English from Binance.com Academy. The manifest deliberately does not mix
Binance.US or user-generated Square content. Some old Academy URLs redirect; retain both original
and resolved URLs when collecting.

## Corpus schema

Each JSONL row has `doc_id`, `title`, `url`, `original_url`, `language`, `kind`, `fetched_at`,
`source_updated_at` if known, `extractor_version`, `paragraphs`, `content_hash`, and `review_status`.
Each paragraph has `paragraph_id`, `heading`, `text`. The hash is SHA-256 of the canonical JSON
paragraph list (use `io.digest`), not a hash of arbitrary HTML bytes. Chunk metadata refers to this hash.

Official `kind` is `official`; preview `kind` is `synthetic_demo`. These cannot be mixed. Human review
adds `review_status: human_reviewed`, `reviewer`, and `reviewed_at`. The collector never sets these.
Missing collection or update dates stay null; candidate checking is not collection.

Include only stable definitions and conceptual examples. Exclude fees, prices, regional access,
rewards, promotion, account troubleshooting and click-by-click UI navigation. D19/D20 have initial
heading allowlists; review whether current source headings still match. Empty extraction is a failure.

For local imports, extract permitted HTML with `binance_doctalk.ingest.extract_html`, or create the
schema above from legitimately obtained source text. Do not change a third-party note into an
official source by assigning it an official URL. Check exact correspondence to the source page.

## Benchmark design

| Primary category | Smoke | Dev | Test |
|---|---:|---:|---:|
| Ordinary | 2 | 14 | 16 |
| Multiple evidence units | 2 | 6 | 8 |
| Insufficient / outside scope | 2 | 6 | 8 |
| False premise | 1 | 3 | 4 |
| Distractor / instruction interference | 1 | 3 | 4 |
| Total | 8 | 32 | 40 |

Answerable/insufficient counts: smoke 5/3, dev 23/9, test 28/12. These are **design counts**, not
measured performance. False premises can be answerable when evidence allows correction. Unsupported
assertions must not be confidently declared false without evidence.

All questions have IDs, `family_id`, split, category, label, draft reference, required points,
proposed source IDs and review status. `proposed_source_ids` do not constitute gold evidence.

`evidence_groups` is a list of required-fact groups. Each group contains alternative acceptable spans:

```json
[[{"doc_id":"D17","paragraph_id":"p003","start":0,"end":120,"content_hash":"actual hash"}]]
```

Offsets are Python Unicode character positions, start inclusive/end exclusive, in the normalized
paragraph text. The example above is illustrative, not a usable source binding. Each required fact
must have an adequate group; a multiple-evidence case needs multiple groups. Retrieval recall treats
alternatives as alternatives and does not double-count overlapping chunks.

Use `doctalk evidence` to inspect spans. Check the entire selected corpus before declaring a question
unanswerable. Check family assignments and near paraphrases manually; exact duplicate checks do not
detect semantic paraphrases. Shared source documents across splits are expected in this closed-corpus task.

## Publication and rights

Public by default: original code, original question/reference paraphrases and labels, manifest,
configuration, methodology, sanitized predictions and measured aggregates. Do not publish full
source documents, HTML, chunk text, vector indexes or long quotations without verified permission.
Short excerpts are not automatically lawful merely because they are short; check the applicable context.

Review [Binance Terms](https://www.binance.com/en/terms) and the
[Academy disclaimer](https://www.binance.com/en/academy/articles/disclaimer). Public accessibility and
robots permission are not redistribution licences. Check external API processing rights separately.
The repository does not claim permission has been granted.

Local official sources, vector indexes, ledgers and raw contexts belong under ignored `data/local/`
or `artifacts/local/`. Before publication, inspect predictions/grades for long copied material or
unintentionally entered private data. Questions should never contain real account details or keys.

## Versioning and reproducibility

Freeze canonical hashes for source text, chunks/index, benchmark and configurations. Never silently
replace source text under the same version. A changed page requires a new corpus version and renewed
evidence binding. Hashes detect changes but cannot restore content. When an old snapshot cannot be
lawfully shared or recollected, document that exact inference reproduction is unavailable; retained
outputs still allow independent recomputation of aggregate scores.
