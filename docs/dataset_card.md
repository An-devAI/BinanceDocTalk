# Dataset card

## Three separate assets

| Asset | Location | Version/status |
|---|---|---|
| Candidate official-source manifest | `data/manifest.v1.jsonl` | 20 URLs; direct HTTP attempt failed; web-reader snapshot import recorded separately |
| Retrieved source snapshots | Local `corpus.web-extract.v1.jsonl`; public metadata manifest | 20 potentially cached web-reader extracts, 481 paragraphs; human review pending |
| Benchmark | `data/eval/drafts-v2/*.jsonl`; original v1 preserved | 92 AI-authored drafts; whole-corpus gap audits and gold pending |
| Original preview corpus | `data/demo/notes.jsonl` | 12 synthetic notes; not official Binance material |

No training/fine-tuning dataset is used. Never add question/reference-answer files to the index.
All official data is English from Binance.com Academy. The manifest deliberately does not mix
Binance.US or user-generated Square content. Some old Academy URLs redirect; retain both original
and resolved URLs when collecting.

See [collection evidence and review handoff](collection_workflow.md). HTML imports require actual
collection timestamps; a missing corpus is never replaced with synthetic notes under official URLs.

## Curated corpus v2

`data/local/corpus.curated.v2.jsonl` retains 477 paragraphs from the 481-paragraph parent.
`data/curation.v2.json` specifies seven sentence-filter/delete operations with pinned parent
hashes. `docs/corpus_curation_v2.json` records changed paragraph IDs, reasons, lengths and hashes;
`data/corpus.curated.v2.manifest.jsonl` publishes metadata without paragraph text.

Source metadata includes `source_url` (= `url`), `retrieved_at` (= `fetched_at`), `version`,
`content_sha256` (= `content_hash`), parent content hash and source-update evidence. Conflicting
aliases are rejected. Retrieval timestamps are real snapshot retrieval times, not upstream crawl
times or curation dates. Unknown update times are null; unknown retrieval times may be recorded
in a manifest but do not pass the official-index gate. The content hash is SHA-256 over canonical
JSON paragraphs; the raw snapshot hash is separate.

Paragraph IDs survive deletion. Any changed document hash invalidates old evidence bindings,
even for retained paragraphs. Explicit context groups keep selected maker/taker and stop-limit
definition/exception paragraphs together and fail if the unit exceeds the configured token budget.
The market-order execution-certainty wording conflict in D15/D16 is exposed for human adjudication;
source text has not been rewritten to manufacture agreement.

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

## Benchmark v2 design

Current drafts are in `data/eval/drafts-v2/`; original v1 drafts remain unchanged for provenance.
V2 has **92 questions: 10 smoke / 36 dev / 46 test**. Draft label counts are 56 answerable,
24 out-of-scope, and 12 candidate in-scope evidence gaps. These are design counts, not approved
labels or measured performance. All records remain pending human review.

`label` is answerable / insufficient_information / out_of_scope. Independent `test_type` values
are ordinary / multi_evidence / false_premise / distractor / prompt_injection. A false premise
can be corrected with evidence; an injection can be answerable if its remaining request is valid.
An out-of-scope question cites project policy, not Binance evidence about the project's scope.
Policy hashes bind to the actual shared system scope.

Each proposed gap audit enumerates every source document/hash, lexical candidate paragraph IDs,
search terms and proposed missing information. No-hit search is not absence evidence. A human
must inspect all selected text, record absent/partial assessments and notes, and approve the audit.
Finding sufficient evidence requires changing the draft. Gap audits are bound to the corpus hash.
See [the complete protocol](evaluation.md) for review commands, grading and held-out restrictions.

Answerable records require one evidence group per required point; alternatives inside a group
are interchangeable supporting spans. Offsets are Python Unicode character positions, start
inclusive/end exclusive. Proposed source IDs and `candidate_evidence_groups` remain unapproved
suggestions; final `evidence_groups` are empty until actual human binding.

Family IDs and semantic near-paraphrases must be reviewed across splits. Shared source documents
are expected in this closed-corpus task. The assistant has seen drafts; held-out means excluded
from actual tuning rather than independently blind. After tuning on test outputs, a new final
question set with new families and semantic audit is required.

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
