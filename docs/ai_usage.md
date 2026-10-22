# AI-assisted development record

## Initial implementation

An AI coding assistant produced the initial Python modules, optional Streamlit interface,
dependency-free local UI, benchmark drafts, tests and documentation. The owner has not yet
performed the manual source/gold review described in the plan. No owner work, interviews,
API measurements, human annotations or Git history has been fabricated.

The assistant inspected the available environment, consulted official OpenAI API documentation,
ran the local synthetic tests, audited draft counts and exercised offline CLI behavior. See
`verification.md` for actual verification and its limits.

## Decisions to understand and verify personally

1. Why a small local index is sufficient and where cosine/BM25 rankings can fail.
2. Which source paragraphs belong in scope and whether every reference answer follows from them.
3. Why all answer prose belongs in validated claims rather than an uncited summary field.
4. Why a provider/schema failure is not a successful knowledge abstention.
5. What makes the baseline fair and which instructions differ necessarily from RAG.
6. What source, code, prompt and data hashes freeze, and what they cannot reconstruct.
7. How costs, retries and failed reservations are tracked.

Before claiming ownership on a CV, trace source → paragraph → chunk → rank → generation → citation
for several cases; explain the module interfaces; reproduce reports; and independently inspect
the critical paths. Add your actual decisions, corrections and verification outcomes below.

## Owner log — fill with real work

| Date | Work or decision | AI assistance | Owner verification | Evidence |
|---|---|---|---|---|
| Pending | Source and benchmark review | Drafting/extraction assistance | Not completed | — |

Do not invent bugs the AI supposedly made or backdate commits. Record corrections only when they
actually occur, and retain the relevant diff/test evidence.

## Actual assistant verification corrections

- A multiline context-manager syntax error in a transport test was caught by test discovery and
  corrected before the passing suite. This was not a production API finding.
- The initial credential check covered legacy Kaggle token locations but missed the existing OAuth
  credentials file. Checking the official cached SDK path found that configuration, and the helper
  now reuses an unexpired access token without printing it or requesting another token.
- `find_spec('kaggle')` alone incorrectly treated the project's `kaggle/` namespace directory as an
  installed CLI. Detection now requires a real package origin/distribution, with a regression test.

- Kaggle CLI 2.2.4 returned exit code zero while printing a server-side maximum GPU-session
  rejection. The helper now requires the explicit successful-version acknowledgement and tests
  both rejection and acceptance; an accepted process exit alone is not a remote-run result.
- The indexed web-extract importer initially interpreted an indented code comment as a second
  article title. The import failed without creating a corpus; code lines are now excluded before
  heading parsing, with a synthetic regression test.

These were assistant-discovered corrections. They are not claims of personal human debugging work.
