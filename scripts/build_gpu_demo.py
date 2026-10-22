"""Render actual saved Qwen outputs; no generation, grading, or browser screenshot fabrication."""
from __future__ import annotations
from html import escape
from io import BytesIO
import json
from pathlib import Path
import sys
import textwrap
from build_demo import avi

ROOT=Path(__file__).resolve().parents[1]


def main(source: Path):
    run=json.loads(source.read_text())
    if run.get('errors') != 0 or run.get('kind') != 'synthetic_integration_only':
        raise ValueError('Use a completed, error-free synthetic integration run')
    rows=run['predictions']
    if len(rows)!=6 or any(r['status']=='error' for r in rows):
        raise ValueError('Preserve all six outputs; do not select only successful predictions')
    out=ROOT/'demo'
    (out/'gpu-recorded-results.json').write_text(json.dumps(run,indent=2)+'\n')
    notes={}
    for row in [json.loads(l) for l in (ROOT/'data/demo/notes.jsonl').read_text().splitlines()]:
        notes[row['doc_id']+':c0001']='\n\n'.join(p['text'] for p in row['paragraphs'])
    def card(prediction):
        content='<p class="status">'+escape(prediction['status'].upper())+' · '+str(round(prediction['latency_ms']))+' ms observed</p>'
        if prediction.get('message'):content+='<p>'+escape(prediction['message'])+'</p>'
        for claim in prediction.get('claims',[]):
            content+='<p>'+escape(claim['text'])+'</p><small>'+escape(', '.join(claim['evidence_ids']) or 'No retrieval citations — baseline')+'</small>'
        ids=sorted({i for c in prediction.get('claims',[]) for i in c['evidence_ids']})
        for cid in ids:
            if cid not in notes:raise ValueError('Unknown original-note citation')
            content+='<details><summary>'+escape(cid)+' · original note</summary><p>'+escape(notes[cid])+'</p></details>'
        return '<section><h3>'+escape(prediction['mode'].upper())+'</h3>'+content+'</section>'
    questions=list(dict.fromkeys(r['question'] for r in rows))
    blocks=[]
    for q in questions:
        cases=[r for r in rows if r['question']==q]
        assert {r['mode'] for r in cases}=={'baseline','rag'}
        blocks.append('<article><h2>'+escape(q)+'</h2><div class="grid">'+''.join(card(r) for r in cases)+'</div></article>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>BinanceDocTalk — recorded Qwen GPU outputs</title><style>
body{background:#f4f5ee;color:#173c31;font:17px/1.6 system-ui;max-width:1180px;margin:40px auto;padding:0 24px}header{padding:20px 0}h1{font-size:40px;line-height:1.15}h2{font-size:24px}h3{letter-spacing:.08em;font-size:15px}.tag{font-size:13px;letter-spacing:.09em}article{margin:36px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{border:1px solid #bac9bc;background:white;padding:24px;border-radius:14px}.status{color:#38654c;font-size:14px}small{color:#666}details{margin-top:18px;border-top:1px solid #ddd;padding-top:12px}summary{cursor:pointer;font-size:14px}footer{border-top:1px solid #bac9bc;padding:24px 0;font-size:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}h1{font-size:32px}}</style><header><div class="tag">RECORDED INFERENCE · ORIGINAL SYNTHETIC NOTES</div><h1>BinanceDocTalk<br>Qwen on Kaggle GPU</h1><p>Actual saved outputs from three integration questions, both without retrieval and with RAG. This page does not run a model.</p><p><b>JSON/citation structure validated; factual correctness and semantic citation support are ungraded.</b> These six outputs are not a held-out benchmark. Inspect all outputs and the original notes below.</p></header>'''
    html+=''.join(blocks)
    html+='<footer>Qwen3-4B-Instruct-2507 · '+escape(run['runtime']['gpu'])+' · FP16 · Source SHA-256: '+escape(run['source_sha256'])+'<br>API spend: $0; GPU compute is not priced here. Independent educational project, not affiliated with Binance. Official-source/gold review and quality metrics remain pending.<br><a href="gpu-recorded-results.json">Full recorded JSON</a> · <a href="../docs/error_analysis.md">Observed limitations and failure analysis</a></footer></html>'
    (out/'gpu-recorded-preview.html').write_text(html)
    from PIL import Image,ImageDraw,ImageFont
    path='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    title=ImageFont.truetype(path,36);regular=ImageFont.truetype(path,22);small=ImageFont.truetype(path,17)
    def base(subtitle):
        image=Image.new('RGB',(1280,720),'#14241e');draw=ImageDraw.Draw(image)
        draw.text((55,35),'BinanceDocTalk — Qwen GPU',fill='#ece9c7',font=title)
        draw.text((55,95),'RECORDED MODEL OUTPUTS | ORIGINAL NOTES | NOT LIVE',fill='#afd5b5',font=small)
        draw.rounded_rectangle((40,140,1240,650),radius=16,fill='#1e342b')
        draw.text((65,165),subtitle,fill='#e8d990',font=regular)
        draw.text((55,678),'Independent project · No official-source quality score · Human review remains pending',fill='#b0c3b3',font=small)
        return image,draw
    intro,draw=base('ACTUAL KAGGLE INTEGRATION — SIX OUTPUTS RETAINED')
    intro_lines=['Qwen3-4B-Instruct-2507 + BGE-small-en-v1.5',
        f"Runtime: {run['runtime']['gpu']} / FP16 / CUDA {run['runtime']['cuda']}",
        'Baseline and RAG share one loaded model.',
        'JSON Schema constraints + application citation validation.',
        'No paid model API calls. No training. No trading execution.',
        'Structure passed. Answer correctness is not yet graded.']
    for i,line in enumerate(intro_lines):draw.text((65,245+i*49),line,fill='#d7e3d8',font=regular)
    images=[intro]
    for prediction in [r for r in rows if r['mode']=='rag'][1:]:
        image,draw=base('RAG · '+prediction['status'].upper()+' · '+str(round(prediction['latency_ms']))+' ms observed')
        y=220
        for line in textwrap.wrap(prediction['question'],82):
            draw.text((65,y),line,fill='#ffffff',font=regular);y+=30
        y+=24
        content=' '.join(c['text'] for c in prediction.get('claims',[])) or prediction['message']
        wrapped=textwrap.wrap(content,82)
        if len(wrapped)>10:raise ValueError('Recorded text needs a larger video layout; do not silently truncate')
        for line in wrapped:draw.text((65,y),line,fill='#d7e3d8',font=regular);y+=30
        ids=sorted({i for c in prediction.get('claims',[]) for i in c['evidence_ids']})
        if ids:draw.text((65,600),'Original-note citations: '+', '.join(ids),fill='#e8d990',font=small)
        images.append(image)
    frames=[]
    for i,image in enumerate(images):
        image.save(ROOT/f'artifacts/local/gpu-demo-scene-{i}.png')
        buffer=BytesIO();image.save(buffer,format='JPEG',quality=90)
        frames.extend([buffer.getvalue()]*10)
    images[0].save(out/'gpu-preview.png')
    (ROOT/'artifacts/local/gpu-walkthrough.avi').write_bytes(avi(frames,1280,720))
    print('Rendered all six actual outputs in HTML/JSON and a 30-second recorded GPU walkthrough.')


if __name__=='__main__':main(Path(sys.argv[1]))
