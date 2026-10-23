"""Prepare a private, smoke-only Kaggle input bundle and notebook. No network calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import digest,file_hash,read_jsonl,write_json
from kaggle_task import cell,prepare_gpu
from official_smoke import verify_inputs


def prepare_inputs(owner,corpus,questions,decision,output):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}',owner):raise ValueError('Invalid Kaggle owner slug')
    if output.exists():raise ValueError('Private input bundle exists; preserve it')
    docs=load_documents(corpus);rows=read_jsonl(questions)
    if len(docs)!=20 or len(rows)!=10 or any(r['split']!='smoke' for r in rows):
        raise ValueError('Only 20-source/10-smoke inputs are allowed; no development or test upload')
    validate_questions(rows,True,docs)
    inputs={'corpus.reviewed.v2.jsonl':corpus,'smoke.reviewed.v2.jsonl':questions,'owner-signoff.json':decision}
    hashes={name:file_hash(path) for name,path in inputs.items()}
    slug='binancedoctalk-smoke-inputs-'+digest(hashes)[:12]
    ref=owner+'/'+slug
    output.mkdir(parents=True)
    for name,path in inputs.items():shutil.copyfile(path,output/name)
    manifest={'kind':'official_source_smoke_only','files':hashes,'source_count':20,'smoke_cases':10,
              'dataset_ref':ref,'private_required':True,'heldout_included':False}
    write_json(output/'input-manifest.json',manifest)
    metadata={'id':ref,'title':'BinanceDocTalk private smoke '+digest(hashes)[:10],
              'isPrivate':True,'licenses':[{'name':'other'}],
              'description':'Private owner-approved educational smoke inputs. Original source copyrights retained. Not a public corpus release; no development/test inference inputs.'}
    write_json(output/'dataset-metadata.json',metadata)
    verify_inputs(output)
    return {'dataset_ref':ref,'input_manifest_sha256':file_hash(output/'input-manifest.json'),
            'private':True,'source_count':20,'smoke_cases':10}


def prepare_notebook(owner,output,dataset_ref,input_manifest_sha256,accelerator='NvidiaTeslaT4'):
    if output.exists():raise ValueError('Notebook folder exists; preserve prior runs')
    if (not re.fullmatch(r'[A-Za-z0-9_-]+/[a-z0-9-]+',dataset_ref)
            or dataset_ref.split('/')[0]!=owner
            or not re.fullmatch(r'[a-f0-9]{64}',input_manifest_sha256)):
        raise ValueError('Use an owned private dataset reference and exact input manifest hash')
    result=prepare_gpu(owner,output,accelerator)
    path=output/'binance_doctalk.ipynb';nb=json.loads(path.read_text())
    nb['cells'][0]=cell('# BinanceDocTalk — official-source smoke only\n\n20 signed-off sources, 10 smoke cases, paired Qwen/BGE pipeline. Private dataset and private notebook. No held-out inference or final quality claim.')
    nb['cells'][-2]=cell('''# Locate only the exact private input manifest; never embed raw source text in notebook code.
expected_input_sha = '''+repr(input_manifest_sha256)+'''
matching = [p for p in Path('/kaggle/input').rglob('input-manifest.json')
            if hashlib.sha256(p.read_bytes()).hexdigest() == expected_input_sha]
assert len(matching) == 1, 'Exactly one matching private smoke input bundle is required'
subprocess.run([sys.executable, 'scripts/check_format_grammar.py'], check=True)
subprocess.run([sys.executable, 'scripts/official_smoke.py', '--inputs', str(matching[0].parent),
                '--output', str(work / 'official-smoke-run')], check=True)
''',True)
    nb['cells'][-1]=cell('Outputs under `official-smoke-run/` retain all 20 predictions including errors. Source text remains in the private input dataset. Model answers are ungraded; no held-out quality metric exists.')
    path.write_text(json.dumps(nb,indent=2)+'\n')
    metadata_path=output/'kernel-metadata.json';meta=json.loads(metadata_path.read_text())
    slug='binancedoctalk-official-smoke-'+result['source_sha256'][:8]
    meta.update(id=owner+'/'+slug,title='BinanceDocTalk Official Smoke '+result['source_sha256'][:8],
                dataset_sources=[dataset_ref],is_private=True)
    write_json(metadata_path,meta)
    return {**result,'kernel':meta['id'],'dataset_ref':dataset_ref,
            'input_manifest_sha256':input_manifest_sha256,'phase':'official_source_smoke_only'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',required=True)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--decision',type=Path,required=True)
    p.add_argument('--input-folder',type=Path,required=True);p.add_argument('--notebook-folder',type=Path,required=True)
    a=p.parse_args()
    inputs=prepare_inputs(a.owner,a.corpus,a.questions,a.decision,a.input_folder)
    notebook=prepare_notebook(a.owner,a.notebook_folder,inputs['dataset_ref'],inputs['input_manifest_sha256'])
    print(json.dumps({'inputs':inputs,'notebook':notebook},indent=2))
