"""Build a complete local annotation packet; suggestions never constitute gold."""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from binance_doctalk.benchmark import scope_policy
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl, utc_now, write_json, write_jsonl
from binance_doctalk.retrieval import Index
from binance_doctalk.schemas import Chunk
from audit_splits import audit

STYLE = 'body{font:16px/1.6 system-ui;max-width:1050px;margin:36px auto;padding:0 20px;color:#17352c}article{border-top:1px solid #bbb;padding:24px 0}summary{cursor:pointer}details{background:#f3f6f4;padding:12px;margin:8px 0}small{color:#555}p,pre{white-space:pre-wrap;overflow-wrap:anywhere}nav{padding:12px;background:#edf2ef}a{color:#165b44}dt{font-weight:bold}section{margin:18px 0}'
NAV = '<nav><a href="sources.html">Whole corpus</a> · <a href="policy.html">Application policy</a> · <a href="smoke.html">Smoke</a> · <a href="dev.html">Development</a> · <a href="test.html">Held-out annotation</a> · <a href="overlap.html">Cross-split audit</a></nav>'


def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+escape(title)+'</title><style>'+STYLE+'</style>'+NAV+'<h1>'+escape(title)+'</h1>'+body+'</html>'


def pretty(value):
    return '<pre>'+escape(json.dumps(value, ensure_ascii=False, indent=2))+'</pre>'


def prepare(output: Path, corpus: Path | None, questions: Path, manifest: Path):
    if output.exists():
        raise ValueError('Review packet exists; preserve previous work and choose a new version')
    docs = load_documents(corpus, allow_unreviewed=True) if corpus else []
    raw_docs = {r['doc_id']: r for r in read_jsonl(corpus)} if corpus else {}
    sources = {r['doc_id']: r for r in read_jsonl(manifest)}
    chunks = [Chunk(f'{d.doc_id}:{p.paragraph_id}', d.doc_id, d.title, d.url, p.heading, p.text,
                    ({'paragraph_id': p.paragraph_id, 'start': 0, 'end': len(p.text)},), d.content_hash)
              for d in docs for p in d.paragraphs]
    index = Index(chunks, {'kind': 'review_suggestions'}) if chunks else None
    output.mkdir(parents=True)
    total = 0
    for split in ['smoke', 'dev', 'test']:
        records, sections = [], []
        for row in read_jsonl(questions / f'{split}.jsonl'):
            suggestions = []
            # Scope decisions and absence claims are not established by top-k document hits.
            for point in row['required_points'] if row['label'] == 'answerable' else []:
                hits = index.search(row['question']+' '+point, 3) if index else []
                suggestions.append({'required_point': point, 'candidates': [
                    {'doc_id': hit['chunk'].doc_id, 'paragraph_id': hit['chunk'].spans[0]['paragraph_id'],
                     'start': 0, 'end': len(hit['chunk'].text), 'content_hash': hit['chunk'].content_hash,
                     'heading': hit['chunk'].heading, 'text': hit['chunk'].text,
                     'score': hit['score'], 'selection_method': 'BM25 suggestion, not gold'} for hit in hits]})
            records.append({'draft': row, 'suggestions': suggestions, 'human_approved': False})
            evidence = '<h3>Gold binding (current record, not inferred)</h3>'+pretty(row.get('evidence_groups', []))
            if row.get('candidate_evidence_groups'):
                evidence += '<h3>Feedback-selected candidates — still unapproved</h3>'+pretty(row['candidate_evidence_groups'])
                for group in row['candidate_evidence_groups']:
                    for ref in group:
                        doc = raw_docs.get(ref['doc_id'])
                        if not doc or doc['content_hash'] != ref['content_hash']:
                            raise ValueError('Feedback candidate belongs to a different corpus version')
                        p = next(p for p in doc['paragraphs'] if p['paragraph_id'] == ref['paragraph_id'])
                        evidence += '<p>'+escape(p['text'][ref['start']:ref['end']])+'</p>'
            for group in suggestions:
                evidence += '<h4>BM25 candidates for: '+escape(group['required_point'])+'</h4>'
                for c in group['candidates']:
                    ref = f"{c['doc_id']}:{c['paragraph_id']}:0:{c['end']}"
                    evidence += '<details><summary>'+escape(ref+' — '+c['heading'])+'</summary><p>'+escape(c['text'])+'</p><a href="sources.html#'+escape(c['doc_id']+'-'+c['paragraph_id'], quote=True)+'">Read surrounding source</a><br><small>Lexical candidate only. Inspect and narrow before binding.</small></details>'
            corpus_audit = row.get('corpus_audit')
            if corpus_audit:
                evidence += '<h3>Whole-corpus gap audit — human review required</h3>'+pretty({k:v for k,v in corpus_audit.items() if k != 'documents'})
                for entry in corpus_audit['documents']:
                    evidence += '<details><summary>'+escape(entry['doc_id']+' — '+entry['assessment'])+'</summary>'+pretty(entry)+'<a href="sources.html#'+escape(entry['doc_id'], quote=True)+'">Read this complete source</a></details>'
            if row.get('policy_basis'):
                evidence += '<h3>Application policy basis (not Binance factual evidence)</h3>'+pretty(row['policy_basis'])
            if not docs:
                evidence += '<p><strong>No corpus collected. Source and absence decisions are blocked.</strong></p>'
            metadata = {k: row.get(k) for k in ['schema_version','split','family_id','test_type','category','label','review_status','reviewer','reviewed_at']}
            sections.append('<article id="'+escape(row['question_id'], quote=True)+'"><h2>'+escape(row['question_id']+' — '+row['question'])+'</h2>'+pretty(metadata)+'<p><b>Draft reference:</b> '+escape(row['reference_answer'])+'</p><h3>Required facts / behavior</h3>'+pretty(row['required_points'])+'<p>'+escape(row.get('behavior_rubric', ''))+'</p>'+evidence+'</article>')
        write_jsonl(output / f'{split}.suggestions.jsonl', records)
        warning = '<p><strong>Draft annotation packet, not approved gold.</strong> All source text is available through Whole corpus. Review type, scope, each required fact and exact spans. Dataset annotation is separate from inference: keep held-out model outputs closed during tuning.</p>'
        (output / f'{split}.html').write_text(page(split.title()+' review', warning+''.join(sections)))
        total += len(records)
    sections = []
    for d in docs:
        raw = raw_docs[d.doc_id]
        metadata = {k:v for k,v in raw.items() if k not in {'paragraphs','paragraph_provenance'}}
        sections.append('<article id="'+escape(d.doc_id, quote=True)+'"><h2>'+escape(d.doc_id+' — '+d.title)+'</h2><a href="'+escape(d.url, quote=True)+'">Official source</a>'+pretty(metadata)+''.join('<section id="'+escape(d.doc_id+'-'+p.paragraph_id, quote=True)+'"><h3>'+escape(p.paragraph_id+' — '+p.heading)+'</h3><p>'+escape(p.text)+'</p></section>' for p in d.paragraphs)+'</article>')
    (output / 'sources.html').write_text(page('Whole source corpus', '<p>All selected source paragraphs, including exceptions and recorded conflicts. Metadata records snapshot retrieval rather than live-page freshness.</p>'+''.join(sections)))
    (output / 'policy.html').write_text(page('Frozen application scope', '<p>This project policy establishes scope and instruction priority. It is separate from source evidence about crypto.</p>'+pretty(scope_policy())))
    try:
        overlap = audit(questions)
    except ValueError as exc:
        overlap = {'status': 'blocked_invalid_benchmark', 'error': str(exc)}
    write_json(output / 'split-overlap.json', overlap)
    (output / 'overlap.html').write_text(page('Cross-split annotation audit', '<p>Lexical ranking is a shortlist, not semantic clearance. Record human decisions separately. Do not use held-out predictions for tuning.</p>'+pretty(overlap)))
    summary = {'created_at': utc_now(), 'documents': len(docs), 'questions': total,
               'human_approved': 0, 'status': 'suggestions_only' if docs else 'blocked_missing_corpus'}
    write_json(output / 'manifest.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/eval/drafts-v2')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/corpus.curated.v2.manifest.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output, args.corpus, args.questions, args.manifest))
