"""Build a local-only, source-linked review of model outputs. Does not assign grades."""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import read_jsonl
from binance_doctalk.benchmark import validate_bound_evidence
from prepare_review import STYLE,pretty


def prepare(corpus,questions,predictions,output):
    if output.exists():raise ValueError('Preserve prior review; choose another output')
    docs={d.doc_id:d for d in load_documents(corpus)}
    rows=read_jsonl(questions);refs={r['question_id']:r for r in rows}
    for row in rows:validate_bound_evidence(row,list(docs.values()))
    seen=set();sections=[]
    def span_text(doc_id,content_hash,span):
        doc=docs[doc_id]
        if doc.content_hash!=content_hash:raise ValueError('Review source hash mismatch')
        p=next(p for p in doc.paragraphs if p.paragraph_id==span['paragraph_id'])
        if not 0<=span['start']<span['end']<=len(p.text):raise ValueError('Review span out of bounds')
        return ('<p>'+escape(p.text[span['start']:span['end']])+'</p><small>'+escape(doc_id+':'+p.paragraph_id)+
                '</small> · <a href="'+escape(doc.url,quote=True)+'">Original source</a>')
    for pred in read_jsonl(predictions):
        qid=pred['question_id'];row=refs[qid];pair=(qid,pred['mode'])
        if pair in seen:raise ValueError('Duplicate output pair')
        seen.add(pair)
        body='<h2>'+escape(qid+' / '+pred['mode'])+'</h2><p>'+escape(row['question'])+'</p>'
        body+='<h3>Reference record — '+escape(row['review_status'])+'</h3>'+pretty(row)
        for group in row.get('evidence_groups',[]):
            body+='<details><summary>Required point evidence alternatives</summary>'
            for ref in group:body+=span_text(ref['doc_id'],ref['content_hash'],ref)
            body+='</details>'
        body+='<h3>Retained model output — not graded</h3>'+pretty(pred)
        for hit in pred.get('retrieved',[]):
            body+='<details><summary>'+escape(hit['chunk_id']+' — retrieved context')+'</summary>'
            for span in hit['spans']:body+=span_text(hit['doc_id'],hit['content_hash'],span)
            body+='</details>'
        sections.append('<article>'+body+'</article>')
    expected={(qid,arm) for qid in refs for arm in ['baseline','rag']}
    if seen!=expected:raise ValueError('Review requires all paired outputs, including failures')
    html='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Private paired-output review</title><style>'+STYLE+'</style><h1>Private paired-output review</h1><p>Source snippets: keep this file local/private. Reference records retain their review status. Outputs are ungraded; inspect correctness, entailment, refusal reason and injection behavior. This viewer records no human judgment.</p>'+''.join(sections)+'</html>'
    output.write_text(html)
    return {'paired_questions':len(refs),'outputs':len(seen),'human_grades_assigned':0}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','questions','predictions','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.corpus,a.questions,a.predictions,a.output),indent=2))
