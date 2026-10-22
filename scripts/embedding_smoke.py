"""Actual CPU BGE/tokenizer integration without loading generation weights or official data."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.ingest import load_documents
from binance_doctalk.provider import create_provider
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config
from binance_doctalk.evaluation import optional_versions
from binance_doctalk.io import utc_now, write_json


def main(output: Path):
    cfg = Config.load(ROOT/'configs/kaggle-rag.toml')
    provider = create_provider(cfg)
    docs = load_documents(ROOT/'data/demo/notes.jsonl')
    begin = time.perf_counter()
    index = build_index(docs,cfg,provider)
    setup_seconds = time.perf_counter()-begin
    assert all(len(v)==384 for v in index.vectors)
    assert all(abs(sum(x*x for x in v)-1)<1e-5 for v in index.vectors)
    questions = ['What is a market order?', 'Does triggering a stop-limit guarantee execution?']
    hits = []
    for q in questions:
        begin = time.perf_counter()
        vector = provider.embed([q],purpose='query')[0]
        result = index.search(q,3,vector)
        hits.append({'question':q,'chunk_ids':[h['chunk'].chunk_id for h in result],
                     'latency_ms':(time.perf_counter()-begin)*1000})
    for chunk in index.chunks:
        for span in chunk.spans:
            doc = next(d for d in docs if d.doc_id==chunk.doc_id)
            paragraph = next(p for p in doc.paragraphs if p.paragraph_id==span['paragraph_id'])
            assert paragraph.text[span['start']:span['end']] in chunk.text
    summary = {'status':'complete','created_at':utc_now(),'kind':'synthetic_embedding_integration_only',
               'source_sha256':os.getenv('DOCTALK_SOURCE_SHA256'),'python':sys.version.split()[0],
               'embedding_model':cfg.embedding_model,'embedding_revision':cfg.embedding_revision,
               'dimensions':384,'chunks':len(index.chunks),'vectors_unit_normalized':True,
               'paragraph_spans_verified':True,'setup_seconds':setup_seconds,
               'installed_versions':optional_versions(),'queries':hits,'generation_executed':False,
               'llm_quality_measured':False}
    write_json(output,summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]))
