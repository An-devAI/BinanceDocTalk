"""Semantic evaluation gates, independent labels/types, and versioned curation."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from binance_doctalk.benchmark import corpus_fingerprint, heldout_identity, scope_policy, validate_v2
from binance_doctalk.chunking import chunk_document
from binance_doctalk.evaluation import make_grade_sheet, score_predictions, validate_questions
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import digest, read_jsonl, write_jsonl
from binance_doctalk.schemas import Document, Paragraph
from test_pipeline import fixture_doc
from curate_corpus import curate
from prepare_benchmark_v2 import census
from prepare_review import prepare
from review_dataset import review_questions


def question(doc, label='answerable', test_type='ordinary'):
    policy = scope_policy()
    return dict(schema_version=2, question_id='FIXTURE', family_id='fixture-family', split='dev',
        question='UNIT TEST ONLY: an illustrative question?', label=label, test_type=test_type,
        reference_answer='UNIT TEST ONLY', required_points=['UNIT TEST FACT'],
        evidence_groups=[[dict(doc_id=doc.doc_id, paragraph_id='p001', start=0, end=5,
                               content_hash=doc.content_hash)]] if label == 'answerable' else [],
        unanswerable_reason='UNIT TEST ONLY' if label != 'answerable' else None,
        review_status='human_reviewed', reviewer='UNIT TEST ONLY', reviewed_at='2026-01-01',
        policy_basis=dict(version=policy['version'], sha256=policy['sha256'],
                          rules=['educational_scope'], rationale='UNIT TEST ONLY'))


class BenchmarkV2Tests(unittest.TestCase):
    def setUp(self):
        self.doc = fixture_doc('official')

    def test_label_and_type_are_independent(self):
        row = question(self.doc, 'out_of_scope', 'prompt_injection')
        self.assertEqual(validate_questions([row], True, [self.doc])['labels'], {'out_of_scope': 1})
        row['label'] = 'prompt_injection'
        with self.assertRaises(ValueError):
            validate_questions([row])
        row.update(label='out_of_scope', test_type='insufficient_information')
        with self.assertRaisesRegex(ValueError, 'test_type'):
            validate_questions([row])

    def test_scope_policy_cannot_be_replaced_by_source_evidence(self):
        row = question(self.doc, 'out_of_scope')
        row.pop('policy_basis')
        with self.assertRaisesRegex(ValueError, 'policy basis'):
            validate_questions([row], True, [self.doc])

    def test_missing_information_requires_complete_human_corpus_audit(self):
        row = question(self.doc, 'insufficient_information')
        row['corpus_audit'] = census([self.doc], ['missing', 'absent'], 'UNIT TEST MISSING FACT')
        with self.assertRaisesRegex(ValueError, 'human review'):
            validate_questions([row], True, [self.doc])
        a = row['corpus_audit']
        a.update(review_status='human_reviewed', reviewer='UNIT TEST ONLY', reviewed_at='2026-01-01')
        a['documents'][0].update(assessment='absent', notes='UNIT TEST ONLY: not in fixture text.')
        validate_questions([row], True, [self.doc])
        a['documents'].append(deepcopy(a['documents'][0]))
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            validate_questions([row], True, [self.doc])
        a['documents'].pop()
        a['documents'][0]['assessment'] = 'sufficient'
        with self.assertRaisesRegex(ValueError, 'absent/partial'):
            validate_questions([row], True, [self.doc])

    def test_audit_and_evidence_must_match_current_source(self):
        row = question(self.doc, 'insufficient_information')
        row['corpus_audit'] = {'corpus_hash': 'stale'}
        with self.assertRaisesRegex(ValueError, 'audit hash'):
            validate_questions([row], True, [self.doc])
        row = question(self.doc)
        row['evidence_groups'][0][0]['content_hash'] = 'stale'
        with self.assertRaisesRegex(ValueError, 'source/hash'):
            validate_questions([row], True, [self.doc])
        row = question(self.doc)
        row['required_points'].append('ANOTHER FACT')
        with self.assertRaisesRegex(ValueError, 'per required fact'):
            validate_questions([row], True, [self.doc])

    def test_helper_does_not_approve_missing_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corpus.jsonl'
            write_jsonl(path, [dict(asdict(self.doc), review_status='human_reviewed',
                                    reviewer='UNIT TEST ONLY', reviewed_at='2026-01-01')])
            row = question(self.doc, 'insufficient_information')
            row['review_status'] = 'pending_human_review'
            with patch('builtins.input', side_effect=['review', 'approve FIXTURE']), patch('builtins.print'):
                reviewed = review_questions([row], 'UNIT TEST ONLY', path)
            self.assertEqual(reviewed[0]['review_status'], 'pending_human_review')

    def test_refusal_reasons_are_not_interchangeable(self):
        rows = [dict(question(self.doc, label), question_id=label, family_id=label,
                     question='UNIT TEST '+label) for label in ['insufficient_information','out_of_scope']]
        predictions = [dict(prediction_id=f'{r["question_id"]}-{arm}', question_id=r['question_id'], mode=arm,
                            status='abstained', reason='out_of_scope', claims=[], message='FIXTURE',
                            latency_ms=1, estimated_usd=0) for r in rows for arm in ['baseline','rag']]
        grades = make_grade_sheet(predictions, rows)
        for g in grades:
            g.update(reviewer='UNIT TEST ONLY', reviewed_at='2026-01-01')
        result = score_predictions(predictions, grades, rows)['systems']['rag']
        self.assertEqual(result['insufficient_evidence_recall'], 0)
        self.assertEqual(result['out_of_scope_recall'], 1)
        self.assertEqual(result['abstention_precision'], .5)
        self.assertEqual(result['response_confusion_matrix']['insufficient_information']['out_of_scope'], 1)
        self.assertIsNone(result['full_accuracy'])
        self.assertIsNone(result['insufficient_evidence_precision'])
        self.assertIsNone(result['prompt_injection']['resistance_rate'])

    def test_injection_can_be_answerable_and_errors_never_pass(self):
        row = question(self.doc, 'answerable', 'prompt_injection')
        predictions = [dict(prediction_id=arm, question_id=row['question_id'], mode=arm,
                            status='answered', reason=None, claims=[{'text':'UNIT TEST','evidence_ids':[]}],
                            message='', latency_ms=1, estimated_usd=0) for arm in ['baseline','rag']]
        grades = make_grade_sheet(predictions, [row])
        for g in grades:
            g.update(reviewer='UNIT TEST ONLY', reviewed_at='2026-01-01', correctness=2)
        with self.assertRaisesRegex(ValueError, 'injection_resisted'):
            score_predictions(predictions, grades, [row])
        for g in grades:
            g['injection_resisted'] = True
        report = score_predictions(predictions, grades, [row])
        self.assertEqual(report['systems']['rag']['prompt_injection']['expected_behavior_success'], 1)
        for p in predictions:
            p.update(status='error', claims=[])
        for g in grades:
            g['correctness'] = 0
        report = score_predictions(predictions, grades, [row])
        self.assertEqual(report['systems']['rag']['prompt_injection']['resistance_rate'], 0)

    def test_heldout_identity_ignores_names_order_and_gold(self):
        rows = [{'question':'A question?'}, {'question':'Another question.'}]
        changed = [dict(r, question_id='renamed', label='out_of_scope') for r in reversed(rows)]
        self.assertEqual(heldout_identity(rows), heldout_identity(changed))

    def test_protected_chunks_keep_exception_and_exact_spans(self):
        paragraphs = (Paragraph('p001','Definition','UNIT TEST: a definition.'),
                      Paragraph('p002','Different heading','UNIT TEST: its exception.'))
        doc = replace(self.doc, paragraphs=paragraphs, context_groups=(('p001','p002'),),
                      content_hash=digest([asdict(p) for p in paragraphs]))
        chunks = chunk_document(doc, size=20, overlap=2, tokenizer='regex_word_v1')
        self.assertEqual(len(chunks), 1)
        self.assertEqual({s['paragraph_id'] for s in chunks[0].spans}, {'p001','p002'})
        for p, span in zip(paragraphs, chunks[0].spans):
            self.assertIn(p.text[span['start']:span['end']], chunks[0].text)
        with self.assertRaisesRegex(ValueError, 'context group exceeds'):
            chunk_document(doc, size=3, overlap=1, tokenizer='regex_word_v1')

    def test_alias_metadata_cannot_contradict_canonical_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'corpus.jsonl'
            write_jsonl(path, [dict(asdict(self.doc), source_url='https://wrong.example')])
            with self.assertRaisesRegex(ValueError, 'Conflicting source metadata'):
                load_documents(path, allow_unreviewed=True)

    def test_curation_preserves_parent_and_paragraph_ids_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            raw = 'L0: Updated Jan 1, 2026\n'
            from hashlib import sha256
            paragraphs = (Paragraph('p001','UNIT','Keep these words. Remove these words.'),
                          Paragraph('p002','UNIT','Remove this whole paragraph.'),
                          Paragraph('p003','UNIT','Keep the final paragraph ID.'))
            doc = replace(self.doc, paragraphs=paragraphs,
                          content_hash=digest([asdict(p) for p in paragraphs]))
            parent = dict(asdict(doc), source_extract_hash=sha256(raw.encode()).hexdigest(),
                          source_updated_at='Jan 1, 2026', paragraph_provenance=[
                              {'paragraph_id':p.paragraph_id,'snapshot_line':i+1} for i,p in enumerate(paragraphs)])
            write_jsonl(home/'parent.jsonl', [parent])
            (home/f'{doc.doc_id}.txt').write_text(raw)
            spec = {'version':'UNIT-v2','operations':[
                dict(doc_id=doc.doc_id, paragraph_id='p001', action='keep_prefix_sentences', count=1,
                     expected_parent_hash=doc.content_hash, reason='UNIT TEST FILTER'),
                dict(doc_id=doc.doc_id, paragraph_id='p002', action='drop',
                     expected_parent_hash=doc.content_hash, reason='UNIT TEST DROP')], 'context_groups':{}}
            (home/'spec.json').write_text(json.dumps(spec))
            before = (home/'parent.jsonl').read_bytes()
            paths = [home/'parent.jsonl',home/'spec.json',home/'out.jsonl',home/'manifest.jsonl',home/'log.json',home]
            result = curate(*paths)
            self.assertEqual((result['paragraphs_before'],result['paragraphs_after']), (3,2))
            self.assertEqual((home/'parent.jsonl').read_bytes(), before)
            row = read_jsonl(home/'out.jsonl')[0]
            self.assertEqual([p['paragraph_id'] for p in row['paragraphs']], ['p001','p003'])
            self.assertEqual(row['paragraphs'][0]['text'], 'Keep these words.')
            self.assertEqual(row['paragraphs'][1]['text'], paragraphs[2].text)
            self.assertEqual(row['parent_content_sha256'], doc.content_hash)
            self.assertNotEqual(row['content_sha256'], doc.content_hash)
            self.assertEqual(row['source_updated_at_evidence']['snapshot_line'], 0)
            self.assertEqual(row['review_status'], 'pending_human_review')
            self.assertNotIn('paragraphs',read_jsonl(home/'manifest.jsonl')[0])
            with self.assertRaisesRegex(ValueError, 'never overwrite'):
                curate(*paths)

    def test_v2_drafts_have_real_gap_candidates_and_no_claimed_gold(self):
        rows = [r for split in ['smoke','dev','test'] for r in read_jsonl(ROOT/f'data/eval/drafts-v2/{split}.jsonl')]
        summary = validate_questions(rows)
        self.assertEqual(summary['splits'], {'smoke':10,'dev':36,'test':46})
        self.assertEqual(summary['labels'], {'answerable':56,'out_of_scope':24,'insufficient_information':12})
        self.assertEqual(summary['reviewed'], 0)
        for r in rows:
            self.assertFalse(r['evidence_groups'])
            if r['label'] == 'insufficient_information':
                self.assertEqual(len(r['corpus_audit']['documents']), 20)
                self.assertTrue(all(d['assessment']=='pending' for d in r['corpus_audit']['documents']))
        case = next(r for r in rows if r['question_id']=='SMOKE-008')
        self.assertEqual((case['label'], case['test_type']), ('out_of_scope','prompt_injection'))


if __name__ == '__main__':
    unittest.main()
