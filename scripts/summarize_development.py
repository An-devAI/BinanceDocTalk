"""Verify retained development outputs and emit metadata only, never a gold quality score."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.io import file_hash,read_jsonl,write_json
from binance_doctalk.schemas import validate_answer


def summarize(folder,questions):
    rows=read_jsonl(questions);by_id={r['question_id']:r for r in rows}
    summary=json.loads((folder/'development-summary.json').read_text())
    manifest=json.loads((folder/'run_manifest.json').read_text())
    predictions=read_jsonl(folder/'predictions.jsonl')
    expected={(qid,mode) for qid in by_id for mode in ['baseline','rag']}
    actual=[(p['question_id'],p['mode']) for p in predictions]
    if len(actual)!=len(expected) or set(actual)!=expected:raise ValueError('Incomplete or duplicate development output pairs')
    if manifest['split']!='dev' or manifest['human_review_complete'] is not False:
        raise ValueError('Expected explicitly unreviewed development run')
    if summary['input_file_hashes']['dev.proposed.v3.jsonl']!=file_hash(questions):raise ValueError('Proposal hash mismatch')
    if any(summary.get(k) is not False for k in ['human_reference_review_complete','human_output_grading_complete','heldout_inference_performed','official_quality_measured','configuration_selected']):
        raise ValueError('Exploration cannot claim human grading, held-out or release selection')
    checked=0
    for pred in predictions:
        if pred['split']!='dev' or pred['question']!=by_id[pred['question_id']]['question']:
            raise ValueError('Output/question mismatch')
        if pred['status']=='error':continue
        answer={k:pred[k] for k in ['status','reason','message','claims']}
        if json.loads(pred['raw_output'])!=answer:raise ValueError('Raw response differs from validated output')
        validate_answer(answer,{h['chunk_id'] for h in pred['retrieved']},pred['mode']=='rag');checked+=1
    pending=read_jsonl(folder/'grades.pending.jsonl')
    if len(pending)!=len(predictions) or any(g.get('reviewer') for g in pending):
        raise ValueError('Expected complete, unreviewed grading template')
    errors=sum(p['status']=='error' for p in predictions)
    if summary['prediction_count']!=len(predictions) or summary['errors']!=errors or manifest['errors']!=errors:
        raise ValueError('Run counts do not match retained outputs')
    latencies={}
    for mode in ['baseline','rag']:
        times=sorted(p['latency_ms'] for p in predictions if p['mode']==mode)
        latencies[mode]={'n':len(times),'p50_ms':statistics.median(times),'p95_ms_nearest_rank':times[math.ceil(.95*len(times))-1]}
    # Explicit allowlist: no raw answers, source paragraphs, contexts or private filesystem paths.
    public={k:summary[k] for k in ['kind','top_k','status','source_sha256','input_manifest_sha256',
        'input_file_hashes','case_count','prediction_count','errors','proposed_labels','index','setup_seconds',
        'runtime','configs','statuses','human_reference_review_complete','human_output_grading_complete',
        'heldout_inference_performed','official_quality_measured','configuration_selected','api_usd','compute_usd','note']}
    public.update(structurally_checked_outputs=checked,latency_descriptive_only=latencies,
        latency_note='One sequential pass per separate GPU worker; includes errors. No cross-worker speed claim.',
        started_at=manifest['started_at'],completed_at=manifest['completed_at'],prompt_hashes=manifest['prompt_hashes'],
        output_sha256={name:file_hash(folder/name) for name in ['predictions.jsonl','run_manifest.json','grades.pending.jsonl','retrieval.provisional.json']},
        retrieval_provisional=json.loads((folder/'retrieval.provisional.json').read_text()))
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve the previous metadata report')
    result=summarize(a.run,a.questions);write_json(a.output,result)
    print(json.dumps({'predictions':result['prediction_count'],'errors':result['errors'],'official_quality_measured':False}))
