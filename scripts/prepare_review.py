"""Build local, read-only review packets; never approve sources or questions."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from html import escape
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl, utc_now, write_json, write_jsonl
from binance_doctalk.retrieval import Index
from binance_doctalk.schemas import Chunk


def prepare(output: Path, corpus: Path | None, questions: Path, manifest: Path):
    if output.exists():
        raise ValueError('Review packet exists; preserve previous work and choose a new version')
    docs = load_documents(corpus, allow_unreviewed=True) if corpus else []
    sources = {r['doc_id']: r for r in read_jsonl(manifest)}
    chunks = [Chunk(f'{d.doc_id}:{p.paragraph_id}', d.doc_id, d.title, d.url, p.heading, p.text,
                    ({'paragraph_id': p.paragraph_id, 'start': 0, 'end': len(p.text)},), d.content_hash)
              for d in docs for p in d.paragraphs]
    index = Index(chunks, {'kind': 'review_suggestions'}) if chunks else None
    output.mkdir(parents=True)
    total = 0
    for split in ['smoke','dev','test']:
        records, sections = [], []
        for row in read_jsonl(questions/f'{split}.jsonl'):
            suggestions = []
            for point in row['required_points']:
                hits = index.search(row['question']+' '+point, 3) if index else []
                suggestions.append({'required_point': point, 'candidates': [
                    {'doc_id': hit['chunk'].doc_id, 'paragraph_id': hit['chunk'].spans[0]['paragraph_id'],
                     'start': 0, 'end': len(hit['chunk'].text), 'content_hash': hit['chunk'].content_hash,
                     'heading': hit['chunk'].heading, 'text': hit['chunk'].text,
                     'score': hit['score'], 'selection_method': 'BM25 suggestion, not gold'} for hit in hits]})
            records.append({'draft': row, 'suggestions': suggestions, 'human_approved': False})
            links = ' · '.join(f'<a href="{escape(sources[d]["url"], quote=True)}">{escape(d)}</a>'
                             for d in row.get('proposed_source_ids',[]) if d in sources)
            evidence = ''
            for group in suggestions:
                evidence += '<h4>Required point: '+escape(group['required_point'])+'</h4>'
                for c in group['candidates']:
                    ref = f"{c['doc_id']}:{c['paragraph_id']}:0:{c['end']}"
                    evidence += '<details><summary>'+escape(ref+' — '+c['heading'])+'</summary><p>'+escape(c['text'])+'</p><small>Suggested paragraph span; inspect and narrow before binding.</small></details>'
            if not docs:
                evidence += '<p><strong>No corpus collected. Open the proposed sources; no evidence binding is available.</strong></p>'
            sections.append('<article><h2>'+escape(row['question_id']+' — '+row['question'])+'</h2><p>Draft label: '+escape(row['label'])+' · '+links+'</p><p><b>Draft reference:</b> '+escape(row['reference_answer'])+'</p>'+evidence+'</article>')
        write_jsonl(output/f'{split}.suggestions.jsonl', records)
        html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BinanceDocTalk '+split+' review</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:40px auto;padding:0 20px;color:#17352c}article{border-top:1px solid #bbb;padding:24px 0}summary{cursor:pointer}details{background:#f3f6f4;padding:12px;margin:8px 0}small{color:#555}p{white-space:pre-wrap}</style><h1>'+split.title()+' review packet</h1><p>Read-only AI-assisted suggestions. No source or gold approval. Correct labels, required points and exact evidence using the review helper. Keep test output closed during system tuning; dataset annotation is separate from inference.</p>'+''.join(sections)+'</html>'
        (output/f'{split}.html').write_text(html)
        total += len(records)
    (output/'sources.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Source review</title><h1>Pending source review</h1>'+''.join('<article><h2>'+escape(d.doc_id+' — '+d.title)+'</h2><p>'+escape(d.url)+'</p>'+''.join('<h3>'+escape(p.paragraph_id+' — '+p.heading)+'</h3><p>'+escape(p.text)+'</p>' for p in d.paragraphs)+'</article>' for d in docs)+'</html>')
    summary = {'created_at': utc_now(), 'documents': len(docs), 'questions': total,
               'human_approved': 0, 'status': 'suggestions_only' if docs else 'blocked_missing_corpus'}
    write_json(output/'manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--questions', type=Path, default=ROOT/'data/eval/drafts')
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/manifest.v1.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.corpus, args.questions, args.manifest))
