"""Development proposals are executable diagnostics, never implicitly approved gold."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from prepare_development_experiment import prepare_inputs,prepare_notebook
from development_experiment import verify_inputs
from prepare_development_proposal import validate_proposal
from binance_doctalk.benchmark import validate_bound_evidence
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.io import read_jsonl,write_jsonl,write_json,file_hash
from test_pipeline import fixture_doc


def fixture(home):
    source=fixture_doc('official')
    docs=[dict(asdict(source),doc_id=f'UNIT-{i:02}',review_status='human_reviewed',reviewer='UNIT TEST ONLY',reviewed_at='2026-01-01T00:00:00+00:00') for i in range(20)]
    rows=[dict(schema_version=2,question_id=f'UNIT-DEV-{i:02}',family_id=f'unit-{i}',split='dev',
        question=f'UNIT TEST ONLY question {i}?',label='answerable',test_type='ordinary',
        reference_answer='UNIT TEST ONLY',required_points=['alpha'],
        evidence_groups=[[dict(doc_id='UNIT-00',paragraph_id='p001',start=0,end=5,content_hash=source.content_hash)]],
        review_status='ai_annotated_pending_human_review',annotation_author='AI assistant',
        reviewer=None,reviewed_at=None) for i in range(36)]
    corpus=home/'source.jsonl';proposal=home/'proposal.jsonl';changes=home/'changes.json'
    write_jsonl(corpus,docs);write_jsonl(proposal,rows)
    write_json(changes,dict(proposal_sha256=file_hash(proposal),corpus_file_sha256=file_hash(corpus)))
    return corpus,proposal,changes


class DevelopmentTests(unittest.TestCase):
    def test_bound_geometry_is_independent_of_human_review(self):
        doc=fixture_doc('official')
        row=dict(question_id='UNIT',label='answerable',required_points=['alpha'],evidence_groups=[[
            dict(doc_id=doc.doc_id,paragraph_id='p001',start=0,end=5,content_hash=doc.content_hash)]])
        validate_bound_evidence(row,[doc])
        bad=deepcopy(row);bad['evidence_groups'][0][0]['end']=999
        with self.assertRaisesRegex(ValueError,'character span'):validate_bound_evidence(bad,[doc])
        bad=deepcopy(row);bad['evidence_groups'][0][0]['content_hash']='bad'
        with self.assertRaisesRegex(ValueError,'source/hash'):validate_bound_evidence(bad,[doc])

    def test_private_development_inputs_reject_scope_and_integrity_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);args=fixture(home)
            result=prepare_inputs('unit-owner',*args,home/'bundle')
            docs,rows,manifest=verify_inputs(home/'bundle')
            self.assertTrue(result['private']);self.assertEqual((len(docs),len(rows)),(20,36))
            with self.assertRaisesRegex(ValueError,'human review is incomplete'):
                validate_questions(rows,True,docs)
            path=home/'bundle/input-manifest.json'
            for flag in ['private_required','heldout_included','human_gold_approved']:
                bad=deepcopy(manifest);bad[flag]=not manifest[flag];write_json(path,bad)
                with self.assertRaisesRegex(ValueError,'explicitly unreviewed'):verify_inputs(home/'bundle')
            write_json(path,manifest)
            extra=home/'bundle/test.jsonl';extra.write_text('UNIT TEST ONLY')
            with self.assertRaisesRegex(ValueError,'Unexpected files'):verify_inputs(home/'bundle')
            extra.unlink()
            proposal=home/'bundle/dev.proposed.v3.jsonl';proposal.write_text(proposal.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_inputs(home/'bundle')

    def test_proposals_cannot_acquire_reviewers_or_include_test_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));rows=read_jsonl(args[1]);doc=fixture_doc('official')
            rows[0]['reviewer']='UNIT TEST HUMAN'
            with self.assertRaisesRegex(ValueError,'human reviewer'):validate_proposal(rows,[doc])
            rows[0]['reviewer']=None;rows[0]['split']='test'
            with self.assertRaisesRegex(ValueError,'held-out is forbidden'):validate_proposal(rows,[doc])

    def test_notebooks_are_private_compile_and_run_development_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            for k in [3,5]:
                folder=Path(tmp)/f'k{k}'
                result=prepare_notebook('unit-owner',folder,'unit-owner/unit-private-dev','a'*64,k)
                meta=json.loads((folder/'kernel-metadata.json').read_text())
                self.assertTrue(meta['is_private']);self.assertEqual(meta['dataset_sources'],['unit-owner/unit-private-dev'])
                nb=json.loads((folder/'binance_doctalk.ipynb').read_text())
                for i,c in enumerate(nb['cells']):
                    if c['cell_type']=='code':compile(''.join(c['source']),f'dev-{k}-{i}','exec')
                final=''.join(nb['cells'][-2]['source'])
                self.assertIn('scripts/development_experiment.py',final)
                self.assertNotIn('scripts/official_smoke.py',final)
                self.assertEqual(result['top_k'],k)

    def test_public_proposals_keep_authorship_and_source_based_revisions(self):
        rows=read_jsonl(ROOT/'data/eval/proposals/dev.v3.jsonl')
        self.assertEqual(len(rows),36)
        self.assertTrue(all(r['review_status']=='ai_annotated_pending_human_review' and not r['reviewer'] for r in rows))
        for qid in ['DEV-035','DEV-036']:
            row=next(r for r in rows if r['question_id']==qid)
            self.assertEqual(row['label'],'answerable');self.assertTrue(row['evidence_groups'])
        gaps=[r for r in rows if r['label']=='insufficient_information']
        self.assertEqual(len(gaps),2)
        self.assertTrue(all(len(r['corpus_audit']['documents'])==20 for r in gaps))
        parents=read_jsonl(ROOT/'data/eval/drafts-v2/dev.jsonl')
        self.assertTrue(all(r['review_status']=='pending_human_review' for r in parents))
        self.assertTrue(all(next(r for r in parents if r['question_id']==qid)['label']=='insufficient_information'
            for qid in ['DEV-035','DEV-036']))

if __name__=='__main__':unittest.main()
