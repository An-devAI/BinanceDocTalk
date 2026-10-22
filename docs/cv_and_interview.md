# CV and interview notes

Do not present pending official-source measurements as completed work. The initial code was
AI-assisted; personal review, decisions and experiments must be real and explainable.

## Applied Data Scientist emphasis

Lead with grouped benchmark construction, manual evidence verification, paired baselines,
metrics/uncertainty, dev-only selection and failure analysis. Discuss limitations of self-authored
small datasets and corpus-relative answerability.

## Applied AI Application Engineer emphasis

Lead with the working pipeline, interfaces, citation/schema validation, error handling, budgets,
tests, local reproducibility and latency/cost tradeoffs. Explain why more infrastructure was unnecessary.

The target role descriptions consulted during planning:
[Applied Data Scientist](https://jobs.lever.co/binance/ae1a07c1-c971-403c-906b-79d8c16e4f2d) and
[Applied AI Application Engineer](https://jobs.lever.co/binance/ef77ec2f-1ffa-49bc-a18d-9f019ca1edae).
Vacancies and requirements can change; check again before applying.

## Technical milestone that can be stated now

“Implemented a configurable RAG application with pinned local-model backends, provenance checks,
and paired evaluation safeguards; verified local Qwen/BGE retrieval and inference on Kaggle GPU, and maintained passing CI
on Python 3.11–3.13.” Describe the 80 questions as unreviewed drafts until actual annotation.
The GPU integration evidence includes 67 passing tests and six structurally validated original-note
outputs; these are not accuracy or semantic citation-support measurements.

## Bullets to use only after completing the corresponding work

- Built an evidence-grounded crypto knowledge assistant over [N_sources] official Binance sources,
  with passage-level citations, evidence-based abstention, and a reproducible local demo.
- Designed and manually verified an [N_questions]-question benchmark with grouped smoke, development,
  and held-out splits; compared a no-retrieval LLM with RAG, measuring [B]% versus [R]% full-answer
  accuracy on [N_test] held-out questions.
- Implemented versioned ingestion, retrieval, citation validation, and automated tests; measured
  [P50]/[P95] s latency and [$C] per query, documenting failures and verification of AI-assisted code.

## Questions to answer without relying on AI

1. Why RAG rather than fine-tuning? Why this corpus and language?
2. How are retrieval documents, benchmark gold, and synthetic demo notes isolated?
3. How did you check false-premise labels and corpus-relative insufficiency?
4. Which choices were selected on dev, and what stayed frozen?
5. When did BM25 beat dense retrieval, and why?
6. How do paragraph spans avoid inflated recall from overlapping chunks?
7. Why can a valid citation still fail semantic support?
8. How do false refusal and abstention recall expose trivial refusal strategies?
9. Is the no-retrieval baseline fair? Did it ever outperform RAG?
10. What can 28 answerable and 12 insufficient test cases establish?
11. Why is a triggered stop-limit not necessarily filled?
12. What dominates end-to-end latency and cost, including failures?
13. Which results can be recomputed when the provider or page changes?
14. What did you personally check, and which changes were AI-assisted?
15. What next experiment follows from the observed errors?
