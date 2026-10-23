"""AI-only development judging, with a preregistered pilot gate. Never consumes held-out."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import tomllib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.ai_judge import (PROMPT,pilot_cases,pilot_match,reference_payload,output_payload,
    perturbation_sample,aggregate_ai,reconstruct)
from binance_doctalk.benchmark import scope_policy,validate_bound_evidence,corpus_fingerprint
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import digest,file_hash,read_jsonl,utc_now,write_json,write_jsonl
from binance_doctalk.schemas import validate_answer
from binance_doctalk.local_judge import LocalJudge,BudgetStop,validate_config
from prepare_development_proposal import validate_proposal

BASE={'corpus.reviewed.v2.jsonl','dev.proposed.v3.jsonl','proposal-manifest.json'}
FILES=BASE|{f'k{k}.{name}.json' for k in [3,5] for name in ['run_manifest','development-summary']}|{f'k{k}.predictions.jsonl' for k in [3,5]}


def verify_inputs(folder):
    manifest=json.loads((folder/'input-manifest.json').read_text())
    if (manifest.get('kind')!='ai_only_dev_judging' or set(manifest.get('files',{}))!=FILES
            or manifest.get('private_required') is not True or manifest.get('heldout_included') is not False
            or manifest.get('human_gold_approved') is not False):raise ValueError('Only private AI-only development inputs are accepted')
    if {p.name for p in folder.iterdir()}-FILES-{'input-manifest.json','dataset-metadata.json'}:
        raise ValueError('Unexpected input files; no held-out inputs allowed')
    for name,h in manifest['files'].items():
        if file_hash(folder/name)!=h:raise ValueError('Judge input hash mismatch')
    docs=load_documents(folder/'corpus.reviewed.v2.jsonl');rows=read_jsonl(folder/'dev.proposed.v3.jsonl')
    if len(docs)!=20:raise ValueError('Expected 20 sources')
    validate_proposal(rows,docs);by_id={d.doc_id:d for d in docs};questions={r['question_id']:r for r in rows}
    changes=json.loads((folder/'proposal-manifest.json').read_text())
    if changes['proposal_sha256']!=file_hash(folder/'dev.proposed.v3.jsonl') or changes['corpus_file_sha256']!=file_hash(folder/'corpus.reviewed.v2.jsonl'):
        raise ValueError('Proposal belongs to different corpus or annotations')
    predictions=[]
    for k in [3,5]:
        run=json.loads((folder/f'k{k}.run_manifest.json').read_text());summary=json.loads((folder/f'k{k}.development-summary.json').read_text())
        if (run['split']!='dev' or run['human_review_complete'] is not False or run['question_hash']!=digest(rows)
                or run['corpus']['corpus_hash']!=corpus_fingerprint(docs) or summary['top_k']!=k
                or summary['heldout_inference_performed'] is not False):raise ValueError('Wrong development run identity')
        arm=read_jsonl(folder/f'k{k}.predictions.jsonl')
        pairs=[(p['question_id'],p['mode']) for p in arm]
        if len(pairs)!=72 or set(pairs)!={(q,m) for q in questions for m in ['baseline','rag']}:
            raise ValueError('Expected all 72 paired outputs per trial')
        for p in arm:
            if p['split']!='dev' or p['question']!=questions[p['question_id']]['question']:raise ValueError('Question mismatch')
            for hit in p['retrieved']:reconstruct(hit,by_id)
            if p['status']!='error':
                value={key:p[key] for key in ['status','reason','message','claims']}
                if json.loads(p['raw_output'])!=value:raise ValueError('Raw response mismatch')
                validate_answer(value,{h['chunk_id'] for h in p['retrieved']},p['mode']=='rag')
            predictions.append(dict(p,trial=k))
    return docs,rows,predictions,manifest


def corpus_windows(doc,judge):
    """Cover every paragraph; copy protected exceptions into windows that need them."""
    units=[{'id':f'{doc.doc_id}:{p.paragraph_id}','text':p.text} for p in doc.paragraphs]
    windows=[];current=[]
    def expand(items):
        ids={u['id'].split(':')[1] for u in items}
        for group in doc.context_groups:
            if ids.intersection(group):ids.update(group)
        return [u for u in units if u['id'].split(':')[1] in ids]
    for unit in units:
        candidate=expand(current+[unit])
        if current and judge.count(candidate)>judge.config['window_tokens']:
            windows.append(expand(current));current=[unit]
        else:current.append(unit)
        if judge.count(expand(current))>judge.config['window_tokens']:
            raise ValueError('Atomic paragraph/context exceeds audit window; no truncation')
    if current:windows.append(expand(current))
    if {u['id'] for w in windows for u in w}!={u['id'] for u in units}:raise ValueError('Incomplete corpus coverage')
    return windows


def run(inputs,output,config,judge_factory=LocalJudge):
    validate_config(config)
    if output.exists():raise ValueError('Preserve prior judge results; choose a new output folder')
    docs,rows,predictions,input_manifest=verify_inputs(inputs)
    output.mkdir(parents=True);by_id={d.doc_id:d for d in docs};policy=scope_policy()
    judge=judge_factory(config,output/'judgments.raw.jsonl')
    manifest={'kind':'ai_only_dev_judging','status':'running','started_at':utc_now(),'config':config,
        'source_sha256':os.getenv('DOCTALK_SOURCE_SHA256'),'input_manifest_sha256':file_hash(inputs/'input-manifest.json'),
        'input_files':input_manifest['files'],'prompt_sha256':digest(PROMPT),'human_review_complete':False,
        'heldout_inference_performed':False,'configuration_selected':False,'official_quality_measured':False,
        'api_usd':0.0,'compute_usd':None,'judge_calls':0}
    write_json(output/'judge-manifest.json',manifest)
    pilot=[];references=[];grades=[];perturbations=[]
    try:
        judge.load()
        for case in pilot_cases():
            record=judge.ask('output',case['payload'])
            pilot.append({'case_id':case['id'],'critical':case['critical'],'expected':case['expected'],
                'task_hash':record['task_hash'],'valid':record['error'] is None,
                'passed':record['error'] is None and pilot_match(record['result'],case['expected'])})
        gate=all(p['valid'] for p in pilot) and sum(p['passed'] for p in pilot)>=config['pilot_min_pass'] and all(p['passed'] for p in pilot if p['critical'])
        write_json(output/'pilot.json',{'passed':gate,'minimum':config['pilot_min_pass'],'cases':pilot,
            'note':'Assistant-authored toy checks, not human calibration or proof of judge reliability.'})
        if not gate:
            manifest['status']='pilot_failed';return manifest
        # Reference eligibility is established and saved BEFORE looking at candidate outputs.
        for row in rows:
            result={'question_id':row['question_id'],'grader_type':'llm','assessment':'unknown',
                    'human_review_complete':False,'task_hashes':[],'flags':row.get('annotation_flags',[])}
            if row['label']=='insufficient_information':
                audits=[];partials=[]
                for doc in docs:
                    for i,window in enumerate(corpus_windows(doc,judge)):
                        payload={'question':row['question'],'scope_policy':policy,'sources':window}
                        rec=judge.ask('audit',payload);result['task_hashes'].append(rec['task_hash'])
                        assessment=rec['result']['assessment'] if rec['result'] else 'unknown'
                        audits.append({'doc_id':doc.doc_id,'content_hash':doc.content_hash,'window':i,
                            'paragraph_ids':[u['id'] for u in window],'assessment':assessment,'task_hash':rec['task_hash']})
                        if assessment=='partial':partials.extend(window)
                states=[a['assessment'] for a in audits]
                combined=None
                if partials and 'sufficient' not in states and 'unknown' not in states:
                    sources=list({u['id']:u for u in partials}.values())
                    rec=judge.ask('audit',{'question':row['question'],'scope_policy':policy,'sources':sources})
                    result['task_hashes'].append(rec['task_hash']);combined=rec['result']['assessment'] if rec['result'] else 'unknown'
                result.update(corpus_audit=audits,combined_partial_assessment=combined,whole_corpus_coverage=True,
                    assessment='disputed' if 'sufficient' in states or combined=='sufficient' else
                        'unknown' if 'unknown' in states or combined=='unknown' else 'consistent')
            else:
                rec=judge.ask('reference',reference_payload(row,by_id,policy));result['task_hashes']=[rec['task_hash']]
                result['assessment']=rec['result']['assessment'] if rec['result'] else 'unknown'
            if result['flags']:
                result['model_assessment']=result['assessment'];result['assessment']='unknown'
            references.append(result)
            write_jsonl(output/'references.ai.jsonl',references)
        write_json(output/'reference-eligibility.json',{'created_before_output_grading':True,'sha256':digest(references),
            'eligible_question_ids':[r['question_id'] for r in references if r['assessment']=='consistent'],
            'human_review_complete':False})
        questions={r['question_id']:r for r in rows}
        for pred in predictions:
            row=questions[pred['question_id']]
            rec=judge.ask('output',output_payload(row,pred,by_id,policy))
            grade={k:pred[k] for k in ['question_id','prediction_id','trial','mode']}
            grade.update(grader_type='llm',label=row['label'],task_hash=rec['task_hash'],judgment=rec['result'],error=rec['error'])
            grades.append(grade);write_jsonl(output/'grades.ai.jsonl',grades)
        for grade in perturbation_sample(grades):
            pred=next(p for p in predictions if p['prediction_id']==grade['prediction_id'] and p['trial']==grade['trial'])
            payload=output_payload(questions[pred['question_id']],pred,by_id,policy)
            payload['reference_sources'].reverse();payload['retrieved_context'].reverse()
            rec=judge.ask('output',payload)
            # Compare substantive labels, not free-text reason wording.
            def labels(value):return {k:v for k,v in value.items() if k!='reason'} if value else None
            perturbations.append({'trial':grade['trial'],'prediction_id':grade['prediction_id'],
                'original_task_hash':grade['task_hash'],'perturbed_task_hash':rec['task_hash'],
                'changed_input':rec['task_hash']!=grade['task_hash'],
                'comparable':grade['judgment'] is not None and rec['result'] is not None,
                'labels_agree':labels(grade['judgment'])==labels(rec['result']) if grade['judgment'] and rec['result'] else None})
        write_json(output/'order-sensitivity.json',{'cases':perturbations,'note':'Input-order sensitivity, NOT accuracy or independent agreement.'})
        write_json(output/'metrics.ai.json',aggregate_ai(grades,references))
        manifest['status']='complete_ai_only'
    except BudgetStop:
        manifest['status']='budget_stopped'
    except Exception as exc:
        manifest.update(status='infrastructure_failed',error=type(exc).__name__+': '+str(exc)[:500])
    finally:
        manifest.update(completed_at=utc_now(),judge_calls=judge.calls,runtime=judge.runtime,
            pilot_cases=len(pilot),reference_records=len(references),output_records=len(grades),
            wall_seconds=time.monotonic()-judge.started)
        pending=[{'kind':'reference','question_id':r['question_id'],'priority':0 if r.get('annotation_flags') or r['label']=='insufficient_information' else 2,
                  'human_review_complete':False} for r in rows]
        pending.extend({'kind':'output','question_id':p['question_id'],'prediction_id':p['prediction_id'],
                        'trial':p['trial'],'human_review_complete':False} for p in predictions)
        write_jsonl(output/'human-review-queue.jsonl',pending)
        manifest['artifact_hashes']={p.name:file_hash(p) for p in output.iterdir() if p.name!='judge-manifest.json' and p.is_file()}
        write_json(output/'judge-manifest.json',manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=ROOT/'configs/kaggle-judge.toml')
    a=p.parse_args();c=tomllib.loads(a.config.read_text());print(json.dumps(run(a.inputs,a.output,c),indent=2))
