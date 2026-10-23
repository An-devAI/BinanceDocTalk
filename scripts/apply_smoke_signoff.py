"""Transcribe an explicit, hash-bound owner decision into the approved smoke inputs only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from binance_doctalk.benchmark import scope_policy
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import file_hash,read_jsonl,write_json,write_jsonl


def apply(decision_path,notes_path,corpus,questions,corpus_output,questions_output,manifest_output):
    if any(p.exists() for p in [corpus_output,questions_output,manifest_output]):
        raise ValueError('Reviewed outputs exist; preserve the previous signoff version')
    decision=json.loads(decision_path.read_text());notes=json.loads(notes_path.read_text())
    if (file_hash(corpus)!=decision['corpus_input_sha256']
            or file_hash(questions)!=decision['smoke_input_sha256']
            or scope_policy()['sha256']!=decision['scope_policy_sha256']
            or notes['owner_decision_sha256']!=file_hash(decision_path)):
        raise ValueError('Owner decision does not bind these exact source/smoke inputs or audit notes')
    if not decision.get('owner_statement') or not decision.get('reviewer'):
        raise ValueError('An explicit owner statement and reviewer identity are required')
    load_documents(corpus,allow_unreviewed=True)
    docs=deepcopy(read_jsonl(corpus));rows=deepcopy(read_jsonl(questions))
    if (len(docs)!=20 or len(rows)!=10 or any(r['split']!='smoke' for r in rows)
            or sorted(d['doc_id'] for d in docs)!=sorted(decision['document_ids'])
            or sorted(r['question_id'] for r in rows)!=sorted(decision['question_ids'])):
        raise ValueError('This decision is limited to the 20-source/10-smoke milestone')
    review={'review_status':'human_reviewed','reviewer':decision['reviewer'],
            'reviewed_at':decision['recorded_at'],'review_method':'owner_chat_signoff',
            'owner_decision_sha256':file_hash(decision_path),
            'review_timestamp_semantics':'Time the owner approval was recorded, not the duration/time of individual source reading.'}
    for doc in docs:doc.update(review)
    for row in rows:
        row.update(review)
        row['review_provenance_note']=decision['review_basis']
        if row['label']=='answerable':
            if not row.get('candidate_evidence_groups'):
                raise ValueError('Do not invent evidence under a blanket signoff')
            row['evidence_groups']=deepcopy(row['candidate_evidence_groups'])
            row['evidence_binding_author']='AI assistant, transcribing the candidates in the approved packet'
        elif row['label']=='insufficient_information':
            supplied=notes['questions'][row['question_id']]
            audit=row['corpus_audit']
            old_refs={e['doc_id']:e['content_hash'] for e in audit['documents']}
            if len(supplied)!=len(old_refs) or {e['doc_id']:e['content_hash'] for e in supplied}!=old_refs:
                raise ValueError('Audit rationales must cover every exact source once')
            by_id={e['doc_id']:e for e in supplied}
            for entry in audit['documents']:entry.update(by_id[entry['doc_id']])
            audit.update(review)
            audit['annotation_author']=notes['annotation_author']
            audit['method']='Owner-declared approval of the whole-corpus smoke label; assistant-attributed per-source rationale and lexical census retained.'
    write_jsonl(corpus_output,docs)
    reviewed_docs=load_documents(corpus_output)
    summary=validate_questions(rows,True,reviewed_docs)
    write_jsonl(questions_output,rows)
    write_jsonl(manifest_output,[{k:v for k,v in d.items() if k!='paragraphs'} for d in docs])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['decision','notes','corpus','questions','corpus-output','questions-output','manifest-output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(apply(a.decision,a.notes,a.corpus,a.questions,a.corpus_output,a.questions_output,a.manifest_output),indent=2))
