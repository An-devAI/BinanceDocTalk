"""Explicit owner signoff transcription and private, smoke-only Kaggle execution boundaries."""
from copy import deepcopy
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from apply_smoke_signoff import apply
from prepare_official_smoke import prepare_inputs,prepare_notebook
from official_smoke import verify_inputs
from binance_doctalk.benchmark import scope_policy
from binance_doctalk.io import file_hash,read_jsonl,write_json,write_jsonl
from binance_doctalk.evaluation import validate_questions,run_experiment
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config
from test_pipeline import fixture_doc


def fixture(home):
    doc=fixture_doc('official')
    docs=[dict(asdict(doc),doc_id=f'UNIT-{i:02}') for i in range(20)]
    rows=[dict(schema_version=2,question_id=f'UNIT-SMOKE-{i:02}',family_id=f'unit-family-{i}',
        split='smoke',question=f'UNIT TEST ONLY question {i}?',label='answerable',test_type='ordinary',
        reference_answer='UNIT TEST ONLY',required_points=['UNIT TEST FACT'],evidence_groups=[],
        candidate_evidence_groups=[[dict(doc_id=docs[0]['doc_id'],paragraph_id='p001',start=0,end=5,
                                          content_hash=doc.content_hash)]],review_status='pending_human_review')
        for i in range(10)]
    corpus,questions=home/'input-corpus.jsonl',home/'input-questions.jsonl'
    write_jsonl(corpus,docs);write_jsonl(questions,rows)
    decision=home/'decision.json';notes=home/'notes.json'
    record=dict(reviewer='UNIT TEST ONLY',owner_statement='UNIT TEST ONLY SIGNOFF',recorded_at='2026-01-01T00:00:00+00:00',
        corpus_input_sha256=file_hash(corpus),smoke_input_sha256=file_hash(questions),
        scope_policy_sha256=scope_policy()['sha256'],document_ids=[d['doc_id'] for d in docs],
        question_ids=[r['question_id'] for r in rows],review_basis='UNIT TEST ONLY; not project quality evidence')
    write_json(decision,record)
    write_json(notes,dict(owner_decision_sha256=file_hash(decision),annotation_author='UNIT TEST ONLY',questions={}))
    return decision,notes,corpus,questions,home/'reviewed-corpus.jsonl',home/'reviewed-smoke.jsonl',home/'manifest.jsonl'


class OfficialSmokeTests(unittest.TestCase):
    def test_signoff_transcribes_only_bound_sources_and_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(Path(folder));before=[p.read_bytes() for p in args[:4]]
            result=apply(*args)
            self.assertEqual(result['reviewed'],10)
            self.assertEqual([p.read_bytes() for p in args[:4]],before)
            for row in read_jsonl(args[5]):
                self.assertEqual(row['evidence_groups'],row['candidate_evidence_groups'])
                self.assertEqual(row['owner_decision_sha256'],file_hash(args[0]))
                self.assertEqual(row['review_method'],'owner_chat_signoff')
            with self.assertRaisesRegex(ValueError,'preserve'):
                apply(*args)

    def test_changed_signoff_inputs_fail_without_reviewed_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(Path(folder));args[3].write_text(args[3].read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'exact source/smoke inputs'):
                apply(*args)
            self.assertFalse(args[4].exists());self.assertFalse(args[5].exists())

    def test_signoff_cannot_expand_to_development(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(Path(folder));rows=read_jsonl(args[3]);rows[0]['split']='dev';write_jsonl(args[3],rows)
            decision=json.loads(args[0].read_text());decision['smoke_input_sha256']=file_hash(args[3]);write_json(args[0],decision)
            notes=json.loads(args[1].read_text());notes['owner_decision_sha256']=file_hash(args[0]);write_json(args[1],notes)
            with self.assertRaisesRegex(ValueError,'limited to'):
                apply(*args)

    def test_private_input_bundle_has_no_development_or_test(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder);args=fixture(home);apply(*args)
            result=prepare_inputs('unit-owner',args[4],args[5],args[0],home/'bundle')
            self.assertTrue(result['private'])
            docs,rows,manifest=verify_inputs(home/'bundle')
            self.assertEqual((len(docs),len(rows)),(20,10))
            self.assertFalse(manifest['heldout_included'])
            metadata=json.loads((home/'bundle/dataset-metadata.json').read_text())
            self.assertTrue(metadata['isPrivate'])
            self.assertEqual(metadata['licenses'],[{'name':'other'}])
            rows[0]['split']='test';write_jsonl(args[5],rows)
            with self.assertRaisesRegex(ValueError,'no development or test upload'):
                prepare_inputs('unit-owner',args[4],args[5],args[0],home/'bad')
            self.assertFalse((home/'bad').exists())

    def test_input_tampering_and_public_flags_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder);args=fixture(home);apply(*args)
            prepare_inputs('unit-owner',args[4],args[5],args[0],home/'bundle')
            manifest_path=home/'bundle/input-manifest.json';manifest=json.loads(manifest_path.read_text())
            write_json(manifest_path,dict(manifest,private_required=False))
            with self.assertRaisesRegex(ValueError,'explicit source/smoke'):
                verify_inputs(home/'bundle')
            write_json(manifest_path,manifest)
            target=home/'bundle/smoke.reviewed.v2.jsonl';target.write_text(target.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                verify_inputs(home/'bundle')

    def test_private_notebook_cells_compile_and_contain_no_corpus_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder)
            result=prepare_notebook('unit-owner',home/'kernel','unit-owner/unit-private-smoke','a'*64)
            meta=json.loads((home/'kernel/kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']);self.assertTrue(meta['enable_gpu'])
            self.assertEqual(meta['dataset_sources'],['unit-owner/unit-private-smoke'])
            nb=json.loads((home/'kernel/binance_doctalk.ipynb').read_text())
            for i,c in enumerate(nb['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'official-smoke-{i}','exec')
            final=''.join(nb['cells'][-2]['source'])
            self.assertIn('scripts/official_smoke.py',final)
            self.assertNotIn('"test"',final)
            self.assertIn('expected_input_sha',final)
            self.assertEqual(result['phase'],'official_source_smoke_only')

    def test_raw_response_is_reset_before_a_query_embedding_failure(self):
        class FixtureProvider:
            def __init__(self):
                self.last_raw_output='STALE PREVIOUS OUTPUT'
                self.usage={};self.retry_count=0;self.actual_models=set()
                self.ledger=SimpleNamespace(read=lambda:{'estimated_usd':0})
            def embed(self,texts,*,purpose='document'):
                if purpose=='query':raise RuntimeError('UNIT TEST QUERY EMBEDDING FAILURE')
                return [[1,0] for _ in texts]
            def generate(self,*args):
                self.last_raw_output='CURRENT UNIT TEST RAW RESPONSE'
                return dict(status='abstained',reason='insufficient_evidence',message='UNIT TEST ONLY',claims=[])
        config=replace(Config(),tokenizer='regex_word_v1')
        index=build_index([fixture_doc('official')],config,FixtureProvider())
        row=dict(schema_version=2,question_id='RAW-FIXTURE',family_id='raw-family',split='smoke',
                 question='UNIT TEST ONLY?',label='answerable',test_type='ordinary',review_status='human_reviewed')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run'
            run_experiment(ROOT,[row],index,[replace(config,system='baseline'),config],lambda c:FixtureProvider(),output,'smoke')
            predictions=read_jsonl(output/'predictions.jsonl')
            rag=next(p for p in predictions if p['mode']=='rag')
            self.assertEqual(rag['status'],'error');self.assertEqual(rag['raw_output'],'')
            baseline=next(p for p in predictions if p['mode']=='baseline')
            self.assertEqual(baseline['raw_output'],'CURRENT UNIT TEST RAW RESPONSE')

    def test_actual_smoke_reference_approvals_do_not_extend_to_dev_test(self):
        rows=read_jsonl(ROOT/'data/eval/v2/smoke.jsonl')
        self.assertEqual(len(rows),10)
        decision=json.loads((ROOT/'data/reviews/owner-smoke-v2.json').read_text())
        self.assertTrue(all(qid.startswith('SMOKE-') for qid in decision['question_ids']))
        self.assertTrue(all(r['review_method']=='owner_chat_signoff' for r in rows))
        for split in ['dev','test']:
            self.assertTrue(all(r['review_status']=='pending_human_review'
                for r in read_jsonl(ROOT/f'data/eval/drafts-v2/{split}.jsonl')))
        for row in rows:
            if row['label']=='insufficient_information':
                self.assertEqual(row['corpus_audit']['annotation_author'],'AI assistant')
                self.assertTrue(all(e['annotation_author']=='AI assistant' for e in row['corpus_audit']['documents']))


if __name__=='__main__':unittest.main()
