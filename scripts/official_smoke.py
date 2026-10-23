"""Run ten signed-off official-source smoke cases; never run or grade held-out here."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.evaluation import run_experiment,validate_questions
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import file_hash,read_jsonl,utc_now,write_json
from binance_doctalk.provider import create_provider
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config


def verify_inputs(folder: Path):
    manifest=json.loads((folder/'input-manifest.json').read_text())
    expected={'corpus.reviewed.v2.jsonl','smoke.reviewed.v2.jsonl','owner-signoff.json'}
    if (manifest.get('kind')!='official_source_smoke_only' or set(manifest.get('files',{}))!=expected
            or manifest.get('private_required') is not True or manifest.get('heldout_included') is not False):
        raise ValueError('Only the explicit source/smoke/signoff input bundle is accepted')
    for name,h in manifest['files'].items():
        if file_hash(folder/name)!=h:raise ValueError('Private smoke input hash mismatch')
    docs=load_documents(folder/'corpus.reviewed.v2.jsonl')
    rows=read_jsonl(folder/'smoke.reviewed.v2.jsonl')
    if len(docs)!=20 or len(rows)!=10 or any(r['split']!='smoke' for r in rows):
        raise ValueError('This runner is restricted to 20 sources and 10 smoke cases')
    validate_questions(rows,True,docs)
    decision_hash=file_hash(folder/'owner-signoff.json')
    if any(r.get('owner_decision_sha256')!=decision_hash for r in [*read_jsonl(folder/'corpus.reviewed.v2.jsonl'),*rows]):
        raise ValueError('Every smoke/source record must refer to the supplied owner decision')
    return docs,rows,manifest


def main(inputs: Path,output: Path):
    if output.exists():raise ValueError('Preserve the earlier smoke run and choose a new output')
    docs,rows,manifest=verify_inputs(inputs)
    configs=[Config.load(ROOT/'configs/kaggle-baseline.toml'),Config.load(ROOT/'configs/kaggle-rag.toml')]
    providers={c.system:create_provider(c) for c in configs}
    started=time.perf_counter()
    providers['rag'].warmup()
    index=build_index(docs,configs[1],providers['rag'],reviewed=True)
    for chunk in index.chunks:
        doc=next(d for d in docs if d.doc_id==chunk.doc_id)
        for span in chunk.spans:
            paragraph=next(p for p in doc.paragraphs if p.paragraph_id==span['paragraph_id'])
            if paragraph.text[span['start']:span['end']] not in chunk.text:
                raise ValueError('Chunk lost exact source provenance')
    setup_seconds=time.perf_counter()-started
    print(json.dumps({'source_count':len(docs),'smoke_cases':len(rows),'dense_chunks':len(index.chunks),
                      'hardware':providers['rag'].runtime_info(),'phase':'smoke_only'},indent=2),flush=True)
    result=run_experiment(ROOT,rows,index,configs,lambda c:providers[c.system],output,'smoke')
    predictions=read_jsonl(output/'predictions.jsonl')
    summary={'created_at':utc_now(),'kind':'official_source_smoke_only',
        'status':'complete' if result['errors']==0 else 'completed_with_errors',
        'source_sha256':os.getenv('DOCTALK_SOURCE_SHA256'),
        'input_manifest_sha256':file_hash(inputs/'input-manifest.json'),
        'owner_decision_sha256':manifest['files']['owner-signoff.json'],
        'input_file_hashes':manifest['files'],'source_count':len(docs),'smoke_cases':len(rows),
        'prediction_count':len(predictions),'errors':result['errors'],
        'index':index.metadata,'setup_seconds':setup_seconds,'runtime':providers['rag'].runtime_info(),
        'configs':[c.to_dict() for c in configs],
        'statuses':{arm:{s:sum(p['mode']==arm and p['status']==s for p in predictions)
                        for s in ['answered','abstained','error']} for arm in ['baseline','rag']},
        'latencies_ms':[{'question_id':p['question_id'],'mode':p['mode'],'latency_ms':p['latency_ms']} for p in predictions],
        'api_usd':0.0,'compute_usd':None,'output_human_grading_complete':False,
        'heldout_inference_performed':False,'heldout_quality_measured':False,
        'note':'Owner-signed-off smoke references; model outputs are ungraded. Timing/status observations are pipeline checks, not final answer/citation quality scores.'}
    write_json(output/'smoke-summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if result['errors']:raise SystemExit('Smoke outputs contain errors; original predictions are retained')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.inputs,a.output)
