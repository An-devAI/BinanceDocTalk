"""Exploratory paired development inference with AI-proposed annotations; never a final score."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.evaluation import run_experiment,retrieval_evaluation,make_grade_sheet
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import file_hash,read_jsonl,utc_now,write_json,write_jsonl
from binance_doctalk.provider import create_provider
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config
from prepare_development_proposal import validate_proposal


def verify_inputs(folder):
    manifest=json.loads((folder/'input-manifest.json').read_text())
    allowed={'corpus.reviewed.v2.jsonl','dev.proposed.v3.jsonl','proposal-manifest.json'}
    if (manifest.get('kind')!='unreviewed_development_exploration'
            or set(manifest.get('files',{}))!=allowed or manifest.get('private_required') is not True
            or manifest.get('heldout_included') is not False or manifest.get('human_gold_approved') is not False):
        raise ValueError('Only an explicitly unreviewed private development bundle is accepted')
    extras={p.name for p in folder.iterdir()}-allowed-{'input-manifest.json','dataset-metadata.json'}
    if extras:raise ValueError('Unexpected files in private development input bundle')
    for name,h in manifest['files'].items():
        if file_hash(folder/name)!=h:raise ValueError('Development input hash mismatch')
    docs=load_documents(folder/'corpus.reviewed.v2.jsonl')
    if len(docs)!=20:raise ValueError('Expected the 20-source signed-off corpus')
    rows=read_jsonl(folder/'dev.proposed.v3.jsonl');validate_proposal(rows,docs)
    proposal=json.loads((folder/'proposal-manifest.json').read_text())
    if (proposal['proposal_sha256']!=manifest['files']['dev.proposed.v3.jsonl']
            or proposal['corpus_file_sha256']!=manifest['files']['corpus.reviewed.v2.jsonl']):
        raise ValueError('Proposal annotations belong to different source inputs')
    return docs,rows,manifest


def main(inputs,output,k):
    if k not in {3,5}:raise ValueError('Only the preregistered k=3/k=5 comparison is supported')
    if output.exists():raise ValueError('Keep prior development predictions; choose another output')
    docs,rows,manifest=verify_inputs(inputs)
    prefix='kaggle' if k==3 else 'kaggle-k5'
    configs=[Config.load(ROOT/f'configs/{prefix}-baseline.toml'),Config.load(ROOT/f'configs/{prefix}-rag.toml')]
    if any(c.top_k!=k for c in configs):raise ValueError('Trial config does not match requested k')
    providers={c.system:create_provider(c) for c in configs}
    started=time.perf_counter();providers['rag'].warmup()
    index=build_index(docs,configs[1],providers['rag'],reviewed=True)
    setup_seconds=time.perf_counter()-started
    print(f'UNREVIEWED DEVELOPMENT EXPLORATION k={k}; 36 cases, 72 paired outputs; no held-out.',flush=True)
    run=run_experiment(ROOT,rows,index,configs,lambda c:providers[c.system],output,'dev',allow_draft=True)
    predictions=read_jsonl(output/'predictions.jsonl')
    if len(predictions)!=72:raise ValueError('Incomplete paired development run; keep original outputs')
    diagnostics=retrieval_evaluation(index,rows,providers['rag'])
    write_json(output/'retrieval.provisional.json',{'kind':'ai_reference_diagnostics_only',
        'human_reference_review_complete':False,'metrics':diagnostics})
    write_jsonl(output/'grades.pending.jsonl',make_grade_sheet(predictions,rows))
    summary={'created_at':utc_now(),'kind':'unreviewed_development_exploration','top_k':k,
        'status':'complete' if not run['errors'] else 'completed_with_errors',
        'source_sha256':os.getenv('DOCTALK_SOURCE_SHA256'),'input_manifest_sha256':file_hash(inputs/'input-manifest.json'),
        'input_file_hashes':manifest['files'],'case_count':36,'prediction_count':72,'errors':run['errors'],
        'proposed_labels':dict(Counter(r['label'] for r in rows)),'index':index.metadata,
        'setup_seconds':setup_seconds,'runtime':providers['rag'].runtime_info(),'configs':[c.to_dict() for c in configs],
        'statuses':{arm:dict(Counter(p['status'] for p in predictions if p['mode']==arm)) for arm in ['baseline','rag']},
        'latencies_ms':[{'question_id':p['question_id'],'mode':p['mode'],'latency_ms':p['latency_ms']} for p in predictions],
        'human_reference_review_complete':False,'human_output_grading_complete':False,
        'heldout_inference_performed':False,'official_quality_measured':False,'configuration_selected':False,
        'api_usd':0.0,'compute_usd':None,
        'note':'AI-proposed development references and ungraded model outputs. Retrieval diagnostics are provisional. Do not report final accuracy, select a release config, or run held-out from this artifact.'}
    write_json(output/'development-summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--k',type=int,choices=[3,5],required=True)
    a=p.parse_args();main(a.inputs,a.output,a.k)
