# Owner review feedback incorporated into v2

The owner inspected all eight original smoke items and identified insufficient review context,
weak BM25 candidates and mixed factual/policy criteria. This record captures requested corrections;
it does not declare source, audit or gold approval.

| Case | Incorporated correction / remaining judgment |
|---|---|
| SMOKE-001 | Candidate alternatives narrowed to the own-funds/no-borrowing sentences in D06:p021 and D07:p011. |
| SMOKE-002 | Direct D15:p006 definition proposed; D15:p034 versus D16:p028 execution-certainty conflict exposed for adjudication. |
| SMOKE-003 | D15:p009 and D09:p014 proposed; independent matching/liquidity points. Type changed to ordinary because one direct passage may suffice. |
| SMOKE-004 | Reference includes multiple price levels and potentially adverse average execution; unrelated OCO is not a gold binding. Candidate sufficiency still needs checking. |
| SMOKE-005 | Current personal account fees classified as out_of_scope under account/current-data policy, rather than used as a true corpus-gap example. |
| SMOKE-006 | Detailed personal futures calculation classified using application scope; missing inputs are not the policy's factual source evidence. |
| SMOKE-007 | Triggering a limit order and absence of a fill guarantee are separate required points; D17:p029 proposed for each. |
| SMOKE-008 | prompt_injection type and out_of_scope label coexist; instruction resistance and expected refusal behavior are scored separately. |

V2 has twelve additional in-scope gap candidates with lexical censuses of all twenty sources.
Their absence/partial judgments and scope rationale require actual human review. The full packet
includes corpus, metadata, test types, split families, policy, audit records and overlap candidates.
Source curation is recorded separately in `corpus_curation_v2.json`; original snapshots and drafts
are retained. No automatic gold decisions, official held-out outputs or quality scores were added.
