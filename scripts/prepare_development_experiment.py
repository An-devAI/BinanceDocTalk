"""Prepare private k=3/k=5 exploration notebooks; proposed annotations remain unreviewed."""
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
from development_experiment import verify_inputs
from prepare_development_proposal import validate_proposal
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl
from prepare_official_smoke import prepare_notebook as prepare_private_notebook
from kaggle_task import cell


def prepare_inputs(owner,corpus,proposal,changes,output):
    if output.exists():raise ValueError('Input bundle exists; preserve the original version')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',owner):raise ValueError('Invalid Kaggle owner')
    docs=load_documents(corpus)
    if len(docs)!=20:raise ValueError('Expected the 20-source signed-off corpus')
    validate_proposal(read_jsonl(proposal),docs)
    files={'corpus.reviewed.v2.jsonl':corpus,'dev.proposed.v3.jsonl':proposal,'proposal-manifest.json':changes}
    hashes={name:file_hash(path) for name,path in files.items()}
    ref=owner+'/binancedoctalk-dev-inputs-'+digest(hashes)[:12]
    output.mkdir(parents=True)
    for name,path in files.items():shutil.copyfile(path,output/name)
    manifest={'kind':'unreviewed_development_exploration','files':hashes,'dataset_ref':ref,
              'private_required':True,'heldout_included':False,'human_gold_approved':False,'development_cases':36}
    write_json(output/'input-manifest.json',manifest)
    verify_inputs(output)
    write_json(output/'dataset-metadata.json',{'id':ref,'title':'BinanceDocTalk private dev '+digest(hashes)[:10],
        'isPrivate':True,'licenses':[{'name':'other'}],
        'description':'Private development exploration only: signed-off source snapshots and AI-proposed development annotations. No held-out inference inputs. Underlying source copyright retained; not a public corpus release.'})
    return {'dataset_ref':ref,'input_manifest_sha256':file_hash(output/'input-manifest.json'),'private':True,'development_cases':36}


def prepare_notebook(owner,output,dataset_ref,input_sha,k):
    if k not in {3,5}:raise ValueError('Only k=3/k=5 development trials are supported')
    result=prepare_private_notebook(owner,output,dataset_ref,input_sha)
    path=output/'binance_doctalk.ipynb';nb=json.loads(path.read_text())
    nb['cells'][0]=cell(f'# BinanceDocTalk — unreviewed development exploration k={k}\n\nAI-proposed annotations, not human gold. Private source inputs; no held-out inference, output grades or release configuration selection.')
    nb['cells'][-2]=cell('''expected_input_sha = '''+repr(input_sha)+'''
matching = [p for p in Path('/kaggle/input').rglob('input-manifest.json')
            if hashlib.sha256(p.read_bytes()).hexdigest() == expected_input_sha]
assert len(matching) == 1, 'Exactly one matching private development input bundle is required'
subprocess.run([sys.executable, 'scripts/check_format_grammar.py'], check=True)
subprocess.run([sys.executable, 'scripts/development_experiment.py', '--inputs', str(matching[0].parent),
                '--output', str(work / 'development-run'), '--k', '''+repr(str(k))+'''], check=True)
''',True)
    nb['cells'][-1]=cell('All 72 paired outputs and blank grade sheets remain in development-run/. Labels and references are AI proposals; do not cite these as official quality results or consume held-out.')
    write_json(path,nb)
    meta_path=output/'kernel-metadata.json';meta=json.loads(meta_path.read_text())
    meta.update(id=f"{owner}/binancedoctalk-dev-k{k}-{result['source_sha256'][:8]}",
                title=f"BinanceDocTalk Dev k{k} {result['source_sha256'][:8]}")
    write_json(meta_path,meta)
    return {**result,'kernel':meta['id'],'top_k':k,'phase':'unreviewed_development_exploration'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--proposal',type=Path,required=True);p.add_argument('--changes',type=Path,required=True)
    p.add_argument('--input-folder',type=Path,required=True);p.add_argument('--notebook-folder',type=Path,required=True)
    a=p.parse_args();inputs=prepare_inputs(a.owner,a.corpus,a.proposal,a.changes,a.input_folder)
    results=[prepare_notebook(a.owner,a.notebook_folder/f'k{k}',inputs['dataset_ref'],inputs['input_manifest_sha256'],k) for k in [3,5]]
    print(json.dumps({'inputs':inputs,'trials':results},indent=2))
