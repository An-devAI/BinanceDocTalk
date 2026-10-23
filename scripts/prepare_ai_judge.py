"""Prepare private saved-output judge inputs and a gated Kaggle notebook. No API calls."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.io import digest,file_hash,write_json
from judge_development import verify_inputs
from prepare_official_smoke import prepare_notebook as private_notebook
from kaggle_task import cell


def prepare_inputs(owner,development_inputs,runs,output):
    if output.exists():raise ValueError('Preserve prior judge inputs')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',owner):raise ValueError('Invalid owner')
    files={name:development_inputs/name for name in ['corpus.reviewed.v2.jsonl','dev.proposed.v3.jsonl','proposal-manifest.json']}
    for k in [3,5]:
        for name in ['predictions.jsonl','run_manifest.json','development-summary.json']:
            files[f'k{k}.{name}']=runs/f'k{k}'/'development-run'/name
    hashes={name:file_hash(p) for name,p in files.items()}
    ref=owner+'/binancedoctalk-ai-judge-inputs-'+digest(hashes)[:12]
    output.mkdir(parents=True)
    for name,path in files.items():shutil.copyfile(path,output/name)
    manifest={'kind':'ai_only_dev_judging','files':hashes,'dataset_ref':ref,'private_required':True,
              'heldout_included':False,'human_gold_approved':False}
    write_json(output/'input-manifest.json',manifest);verify_inputs(output)
    write_json(output/'dataset-metadata.json',{'id':ref,'title':'BinanceDocTalk private AI judge '+digest(hashes)[:8],
        'isPrivate':True,'licenses':[{'name':'other'}],
        'description':'Private source snapshots, AI reference proposals and ungraded development outputs. Not a public source release, human gold or held-out evaluation.'})
    return {'dataset_ref':ref,'input_manifest_sha256':file_hash(output/'input-manifest.json'),'private':True}


def prepare_notebook(owner,output,dataset_ref,input_sha):
    result=private_notebook(owner,output,dataset_ref,input_sha)
    path=output/'binance_doctalk.ipynb';nb=json.loads(path.read_text())
    nb['cells'][0]=cell('# BinanceDocTalk — AI-only saved development judging\n\nPrivate Phi judge, preregistered pilot gate, 3 GPU-hour wall budget. No paid API calls, human signoff, release selection or held-out inference.')
    nb['cells'][-2]=cell('''expected_input_sha = '''+repr(input_sha)+'''
matching = [p for p in Path('/kaggle/input').rglob('input-manifest.json')
            if hashlib.sha256(p.read_bytes()).hexdigest() == expected_input_sha]
assert len(matching) == 1, 'Exactly one matching private AI judge bundle is required'
subprocess.run([sys.executable, 'scripts/judge_development.py', '--inputs', str(matching[0].parent),
                '--output', str(work / 'ai-judge-run')], check=True)
''',True)
    nb['cells'][-1]=cell('A COMPLETE notebook may record pilot_failed or budget_stopped. Inspect ai-judge-run/judge-manifest.json. Only a successful pilot permits development judging; no automatic fallback or gold promotion.')
    write_json(path,nb)
    meta_path=output/'kernel-metadata.json';meta=json.loads(meta_path.read_text())
    meta.update(id=owner+'/binancedoctalk-ai-judge-'+result['source_sha256'][:8],
                title='BinanceDocTalk AI Judge '+result['source_sha256'][:8])
    write_json(meta_path,meta)
    return {**result,'kernel':meta['id'],'phase':'ai_only_dev_judging'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner',required=True)
    for name in ['development-inputs','runs','input-folder','notebook-folder']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();inputs=prepare_inputs(a.owner,a.development_inputs,a.runs,a.input_folder)
    print(json.dumps({'inputs':inputs,'notebook':prepare_notebook(a.owner,a.notebook_folder,inputs['dataset_ref'],inputs['input_manifest_sha256'])},indent=2))
