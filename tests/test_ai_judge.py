"""AI evaluation boundaries, not evidence of real judge accuracy."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import tomllib
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
import test_local_provider as local_fixture
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.ai_judge import (PROMPT,SCHEMAS,pilot_cases,pilot_match,validate_result,
    reference_context,reconstruct,aggregate_ai,perturbation_sample,substantive_labels)
from binance_doctalk.local_judge import LocalJudge,BudgetStop,validate_config
from binance_doctalk.evaluation import score_predictions
from binance_doctalk.web import serve
from binance_doctalk.io import digest,write_json,write_jsonl,read_jsonl
from binance_doctalk.ingest import load_documents
from binance_doctalk.benchmark import corpus_fingerprint
from judge_development import run,verify_inputs,corpus_windows
from prepare_ai_judge import prepare_inputs,prepare_notebook
from ai_report import report
from test_development_experiment import fixture
from test_pipeline import fixture_doc
CONFIG=tomllib.loads((ROOT/'configs/kaggle-judge.toml').read_text())


def expected_result(case):
    e=case['expected']
    return {'correctness':e['correctness'],'refusal_correct':e['refusal_correct'],
        'injection_resisted':e['injection_resisted'],'reason':'UNIT TEST ONLY',
        'citation_support':[{'claim_index':0,'evidence_id':'S1','supports':s} for s in e['supports']]}


def input_fixture(home):
    development=home/'development';development.mkdir();corpus,questions,changes=fixture(development)
    corpus.rename(development/'corpus.reviewed.v2.jsonl');questions.rename(development/'dev.proposed.v3.jsonl');changes.rename(development/'proposal-manifest.json')
    docs=load_documents(development/'corpus.reviewed.v2.jsonl');rows=read_jsonl(development/'dev.proposed.v3.jsonl')
    runs=home/'runs'
    for k in [3,5]:
        out=runs/f'k{k}'/'development-run';out.mkdir(parents=True)
        value=dict(status='abstained',reason='insufficient_evidence',message='UNIT TEST ONLY',claims=[])
        preds=[dict(value,question_id=r['question_id'],question=r['question'],split='dev',mode=m,
                    prediction_id=digest([r['question_id'],m])[:16],retrieved=[],raw_output=json.dumps(value))
               for r in rows for m in ['baseline','rag']]
        write_jsonl(out/'predictions.jsonl',preds)
        write_json(out/'run_manifest.json',dict(split='dev',human_review_complete=False,question_hash=digest(rows),
                   corpus={'corpus_hash':corpus_fingerprint(docs)}))
        write_json(out/'development-summary.json',dict(top_k=k,heldout_inference_performed=False))
    return development,runs


class JudgeContractTests(unittest.TestCase):
    def test_judge_configuration_cannot_unpin_or_relax_the_preregistered_gate(self):
        validate_config(CONFIG)
        for key,value in [('model_revision','main'),('pilot_min_pass',1),('max_gpu_seconds',10801),('seed',True)]:
            with self.assertRaises(ValueError):validate_config(dict(CONFIG,**{key:value}))

    def test_pilot_has_twelve_cases_and_four_critical_checks(self):
        cases=pilot_cases();self.assertEqual(len(cases),12);self.assertEqual(sum(c['critical'] for c in cases),4)
        for case in cases:
            result=expected_result(case);validate_result('output',result,case['payload'])
            self.assertTrue(pilot_match(result,case['expected']))

    def test_missing_citations_and_invalid_correctness_are_not_repaired(self):
        case=next(c for c in pilot_cases() if c['id']=='entailed-citation');result=expected_result(case)
        result['citation_support']=[]
        with self.assertRaisesRegex(ValueError,'citation judgments'):validate_result('output',result,case['payload'])
        case=pilot_cases()[0];result=expected_result(case);result['correctness']=True
        with self.assertRaisesRegex(ValueError,'correctness'):validate_result('output',result,case['payload'])
        case=next(c for c in pilot_cases() if c['id']=='gap-correct');result=expected_result(case);result['correctness']=2
        with self.assertRaisesRegex(ValueError,'must be null'):validate_result('output',result,case['payload'])

    def test_audit_cannot_invent_sources_or_sufficiency_without_evidence(self):
        payload={'sources':[{'id':'S1','text':'UNIT TEST ONLY'}]}
        value=dict(assessment='sufficient',supporting_ids=['FAKE'],reason='UNIT')
        with self.assertRaisesRegex(ValueError,'Invented'):validate_result('audit',value,payload)
        value['supporting_ids']=[]
        with self.assertRaisesRegex(ValueError,'needs evidence'):validate_result('audit',value,payload)

    def test_reference_context_keeps_protected_exception(self):
        doc=replace(fixture_doc('official'),context_groups=(('p001','p003'),))
        row=dict(evidence_groups=[[dict(doc_id=doc.doc_id,paragraph_id='p001',start=0,end=5,content_hash=doc.content_hash)]])
        context=reference_context(row,{doc.doc_id:doc})
        self.assertEqual({x['id'] for x in context},{'TEST:p001','TEST:p003'})

    def test_context_reconstruction_checks_span_and_source(self):
        doc=fixture_doc('official');hit=dict(doc_id=doc.doc_id,content_hash=doc.content_hash,url=doc.url,
             chunk_id='UNIT',spans=[dict(paragraph_id='p001',start=0,end=5)])
        self.assertEqual(reconstruct(hit,{doc.doc_id:doc})['text'],'alpha')
        hit['content_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'source/hash'):reconstruct(hit,{doc.doc_id:doc})

    def test_whole_corpus_windows_preserve_all_paragraphs_and_exceptions(self):
        class Counter:
            config={'window_tokens':100}
            def count(self,items):return sum(len(x['text']) for x in items)
        doc=replace(fixture_doc('official'),context_groups=(('p001','p003'),))
        windows=corpus_windows(doc,Counter())
        self.assertEqual({u['id'] for w in windows for u in w},{'TEST:p001','TEST:p002','TEST:p003'})
        for w in windows:
            if any(u['id']=='TEST:p001' for u in w):self.assertTrue(any(u['id']=='TEST:p003' for u in w))
        Counter.config={'window_tokens':1}
        with self.assertRaisesRegex(ValueError,'no truncation'):corpus_windows(doc,Counter())

    def test_cached_identity_includes_prompt_schema_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            judge=LocalJudge(CONFIG,Path(tmp)/'raw.jsonl');payload=pilot_cases()[0]['payload']
            key=digest(dict(config=CONFIG,prompt=PROMPT,schema=SCHEMAS['output'],kind='output',payload=payload))
            judge.cache[key]={'UNIT':'CACHED'}
            self.assertEqual(judge.ask('output',payload),{'UNIT':'CACHED'});self.assertEqual(judge.calls,0)
            judge.started-=CONFIG['max_gpu_seconds']+1
            with self.assertRaises(BudgetStop):judge.ask('output',payload)

    def test_ai_records_cannot_enter_human_report_or_ui(self):
        with self.assertRaisesRegex(ValueError,'AI grades'):score_predictions([], [{'grader_type':'llm'}], [])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ai.json';write_json(path,dict(status='human_graded',systems={},grader_type='llm'))
            with self.assertRaisesRegex(ValueError,'human-graded'):serve(report_path=path)

    def test_order_check_ignores_citation_pair_order_but_not_support(self):
        a=dict(reason='UNIT A',citation_support=[dict(claim_index=0,evidence_id=i,supports='yes') for i in ['a','b']])
        b=deepcopy(a);b['citation_support'].reverse();b['reason']='UNIT B'
        self.assertEqual(substantive_labels(a),substantive_labels(b))
        b['citation_support'][0]['supports']='no'
        self.assertNotEqual(substantive_labels(a),substantive_labels(b))

    def test_aggregates_keep_unresolved_references_and_denominators(self):
        refs=[dict(question_id='a',assessment='consistent'),dict(question_id='b',assessment='unknown')]
        grades=[dict(question_id=q,trial=3,mode='rag',label='answerable',judgment=dict(correctness=2,
            refusal_correct='not_applicable',injection_resisted='not_applicable',citation_support=[])) for q in ['a','b']]
        result=aggregate_ai(grades,refs);s=result['systems']['k3_rag']
        self.assertEqual(s['outputs_total'],2);self.assertEqual(s['reference_eligible'],1)
        self.assertFalse(result['human_review_complete']);self.assertFalse(result['configuration_selected'])


class JudgePreparationTests(unittest.TestCase):
    def test_private_inputs_accept_only_bound_dev_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);development,runs=input_fixture(home)
            result=prepare_inputs('unit-owner',development,runs,home/'bundle')
            self.assertTrue(result['private']);docs,rows,preds,manifest=verify_inputs(home/'bundle')
            self.assertEqual((len(docs),len(rows),len(preds)),(20,36,144))
            path=home/'bundle/input-manifest.json';bad=deepcopy(manifest);bad['human_gold_approved']=True;write_json(path,bad)
            with self.assertRaisesRegex(ValueError,'AI-only'):verify_inputs(home/'bundle')
            write_json(path,manifest);target=home/'bundle/k3.predictions.jsonl';target.write_text(target.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_inputs(home/'bundle')

    def test_notebook_is_private_compiles_and_uses_exact_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'kernel';prepare_notebook('unit-owner',folder,'unit-owner/unit-private-judge','a'*64)
            meta=json.loads((folder/'kernel-metadata.json').read_text());self.assertTrue(meta['is_private'])
            self.assertEqual(meta['dataset_sources'],['unit-owner/unit-private-judge'])
            nb=json.loads((folder/'binance_doctalk.ipynb').read_text())
            for i,c in enumerate(nb['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'judge-{i}','exec')
            final=''.join(nb['cells'][-2]['source']);self.assertIn('scripts/judge_development.py',final)
            self.assertNotIn('official_smoke.py',final)

    def test_failed_pilot_stops_before_reference_and_output_grading(self):
        class FailingPilot:
            def __init__(self,config,path):self.calls=0;self.runtime={};self.started=time.monotonic()
            def load(self):pass
            def ask(self,kind,payload):
                case=pilot_cases()[self.calls];self.calls+=1
                result=expected_result(case)
                if case['critical']:result['correctness']=2
                return dict(task_hash=str(self.calls),result=result,error=None)
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);development,runs=input_fixture(home);prepare_inputs('unit',development,runs,home/'bundle')
            result=run(home/'bundle',home/'out',CONFIG,FailingPilot)
            self.assertEqual(result['status'],'pilot_failed');self.assertEqual(result['judge_calls'],12)
            self.assertFalse((home/'out/grades.ai.jsonl').exists())
            public=report(home/'out',home/'public.json')
            self.assertIsNone(public['metrics']);self.assertFalse(public['human_review_complete'])
            self.assertEqual(len(read_jsonl(home/'out/human-review-queue.jsonl')),180)

class JudgeIntegrationTests(unittest.TestCase):
    def test_fake_success_orders_references_first_and_reports_ai_only(self):
        class FakeJudge:
            def __init__(self,config,path):
                self.config=config;self.path=path;self.calls=0;self.runtime={'unit_test_only':True}
                self.started=time.monotonic();self.cache={}
            def load(self):pass
            def count(self,value):return 1
            def ask(self,kind,payload):
                key=digest([kind,payload])
                if key in self.cache:return self.cache[key]
                if self.calls<12:result=expected_result(pilot_cases()[self.calls])
                elif kind=='reference':result=dict(assessment='consistent',reason='UNIT TEST ONLY')
                else:
                    assert (self.path.parent/'reference-eligibility.json').exists()
                    result=dict(correctness=0,refusal_correct='no',citation_support=[],
                                injection_resisted='not_applicable',reason='UNIT TEST ONLY')
                record=dict(task_hash=key,grader_type='llm',kind=kind,result=result,error=None,raw_output=json.dumps(result))
                self.cache[key]=record;self.calls+=1
                with self.path.open('a') as h:h.write(json.dumps(record)+'\n')
                return record
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);development,runs=input_fixture(home);prepare_inputs('unit',development,runs,home/'bundle')
            result=run(home/'bundle',home/'out',CONFIG,FakeJudge)
            self.assertEqual(result['status'],'complete_ai_only')
            self.assertEqual(result['reference_records'],36);self.assertEqual(result['output_records'],144)
            public=report(home/'out',home/'public.json',home/'review.html')
            self.assertEqual(public['metrics']['status'],'ai_judged_unvalidated')
            self.assertFalse(public['human_review_complete']);self.assertGreater(public['reused_output_judgments'],0)
            self.assertTrue((home/'review.html').exists())
            with self.assertRaisesRegex(ValueError,'Preserve'):report(home/'out',home/'public.json')

class LocalJudgeTransportTests(unittest.TestCase):
    def test_valid_raw_is_retained_once_and_identical_inputs_are_cached(self):
        case=pilot_cases()[0];value=expected_result(case)
        runtime=local_fixture.LocalProviderTests().fake_runtime(json.dumps(value))
        runtime[2].generation_config=SimpleNamespace(eos_token_id=0)
        runtime[2].generate=Mock(return_value=[[1]*32+[2,0]])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw.jsonl'
            with patch('binance_doctalk.local_judge.generation_runtime',return_value=runtime), patch('binance_doctalk.local_judge.build_prefix_enforcer',return_value=None):
                judge=LocalJudge(CONFIG,path);a=judge.ask('output',case['payload']);b=judge.ask('output',case['payload'])
            self.assertIs(a,b);self.assertEqual(a['result'],value);self.assertEqual(len(read_jsonl(path)),1)
            self.assertFalse(runtime[2].generate.call_args.kwargs['do_sample'])
            self.assertEqual(runtime[2].generate.call_args.kwargs['logits_to_keep'],1)

    def test_prompt_overflow_is_not_truncated_or_generated(self):
        runtime=local_fixture.LocalProviderTests().fake_runtime('{}',input_count=7000);runtime[2].generate=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            with patch('binance_doctalk.local_judge.generation_runtime',return_value=runtime):
                result=LocalJudge(CONFIG,Path(tmp)/'raw.jsonl').ask('output',pilot_cases()[0]['payload'])
            self.assertIsNone(result['result']);self.assertIn('NOT truncated',result['error']);runtime[2].generate.assert_not_called()

    def test_invalid_json_stays_an_error_without_output_repair(self):
        runtime=local_fixture.LocalProviderTests().fake_runtime('```json {} ```');runtime[2].generation_config=SimpleNamespace(eos_token_id=0)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('binance_doctalk.local_judge.generation_runtime',return_value=runtime), patch('binance_doctalk.local_judge.build_prefix_enforcer',return_value=None):
                result=LocalJudge(CONFIG,Path(tmp)/'raw.jsonl').ask('output',pilot_cases()[0]['payload'])
            self.assertIsNone(result['result']);self.assertEqual(result['raw_output'],'```json {} ```')
            self.assertIn('JSONDecodeError',result['error'])

if __name__=='__main__':unittest.main()
