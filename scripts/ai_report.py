"""Verify AI-only judge artifacts; emit metadata and an explicitly unvalidated local review page."""
from __future__ import annotations
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.ai_judge import aggregate_ai,substantive_labels
from binance_doctalk.io import file_hash,read_jsonl,write_json


def report(run,output,html=None):
    if output.exists() or html and html.exists():raise ValueError('Preserve prior AI reports')
    manifest=json.loads((run/'judge-manifest.json').read_text())
    if (manifest['kind']!='ai_only_dev_judging' or any(manifest.get(k) is not False for k in
        ['human_review_complete','heldout_inference_performed','configuration_selected','official_quality_measured'])):
        raise ValueError('Not an explicitly AI-only development run')
    for name,h in manifest['artifact_hashes'].items():
        if Path(name).name!=name or file_hash(run/name)!=h:raise ValueError('Judge artifact hash mismatch')
    pilot=json.loads((run/'pilot.json').read_text()) if (run/'pilot.json').exists() else None
    raw=read_jsonl(run/'judgments.raw.jsonl') if (run/'judgments.raw.jsonl').exists() else []
    for rec in raw:
        if rec['grader_type']!='llm':raise ValueError('Judge authorship mismatch')
        if rec['result'] is not None and json.loads(rec['raw_output'])!=rec['result']:raise ValueError('Judge raw JSON differs')
    public={'kind':'ai_only_dev_judging','status':manifest['status'],'evaluation_status':'ai_judged_unvalidated',
        'grader_type':'llm','human_review_complete':False,'official_quality_measured':False,
        'configuration_selected':False,'heldout_inference_performed':False,
        'config':manifest['config'],'source_sha256':manifest['source_sha256'],
        'input_manifest_sha256':manifest['input_manifest_sha256'],'input_files':manifest['input_files'],
        'prompt_sha256':manifest['prompt_sha256'],'runtime':manifest['runtime'],'started_at':manifest['started_at'],
        'completed_at':manifest['completed_at'],'wall_seconds':manifest['wall_seconds'],
        'judge_calls':manifest['judge_calls'],'reference_records':manifest['reference_records'],
        'output_records':manifest['output_records'],'api_usd':0.0,'compute_usd':None,
        'artifact_hashes':manifest['artifact_hashes'],
        'pilot':{'passed':pilot['passed'],'n':len(pilot['cases']),
            'passed_n':sum(c['passed'] for c in pilot['cases']),'valid_n':sum(c['valid'] for c in pilot['cases']),
            'cases':[{k:c[k] for k in ['case_id','critical','valid','passed']} for c in pilot['cases']],
            'critical_failed_ids':[c['case_id'] for c in pilot['cases'] if c['critical'] and not c['passed']]} if pilot else None,
        'judge_error_count':sum(r['error'] is not None for r in raw),
        'note':'AI-only, not calibrated against human judgments. Pilot checks are assistant-authored toy cases. No human-verified accuracy or held-out result.'}
    if manifest['status']=='complete_ai_only':
        if not pilot or not pilot['passed']:raise ValueError('Development scores require passed pilot')
        references=read_jsonl(run/'references.ai.jsonl');grades=read_jsonl(run/'grades.ai.jsonl')
        if len(references)!=36 or len(grades)!=144:raise ValueError('Incomplete development coverage')
        if any(g.get('grader_type')!='llm' or g.get('reviewer') or g.get('reviewed_at') for g in grades):
            raise ValueError('AI grades cannot declare human review')
        public['metrics']=aggregate_ai(grades,references)
        order=json.loads((run/'order-sensitivity.json').read_text());by_task={r['task_hash']:r for r in raw}
        for case in order['cases']:
            original=by_task[case['original_task_hash']]['result'];changed=by_task[case['perturbed_task_hash']]['result']
            case['labels_agree']=substantive_labels(original)==substantive_labels(changed) if original and changed else None
        order['comparison']='Substantive labels; citation-pair list order and reason wording ignored.'
        public['order_sensitivity']=order
        public['unique_output_judgments']=len({g['task_hash'] for g in grades})
        public['reused_output_judgments']=len(grades)-public['unique_output_judgments']
    else:
        public['metrics']=None
    public['report_code_sha256']={p.name:file_hash(p) for p in [Path(__file__),ROOT/'src/binance_doctalk/ai_judge.py']}
    write_json(output,public)
    if html:
        # This optional page may contain raw model judgments; keep it local/private.
        html.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>AI-only evaluation</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:35px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>LLM-judged — NOT human verified</h1><p>Reference review, semantic grades and release selection remain unapproved. Do not treat judge agreement as truth. This page and raw judgments stay private.</p><h2>Metadata</h2><pre>'+escape(json.dumps(public,indent=2))+'</pre><h2>Retained judgments — no human approval</h2>'+''.join('<details><summary>'+escape(r['kind']+' '+r['task_hash'][:12])+'</summary><pre>'+escape(json.dumps(r,indent=2))+'</pre></details>' for r in raw)+'</html>')
    return public

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--html',type=Path)
    a=p.parse_args();r=report(a.run,a.output,a.html);print(json.dumps({'status':r['status'],'pilot':r['pilot'],'human_review_complete':False},indent=2))
