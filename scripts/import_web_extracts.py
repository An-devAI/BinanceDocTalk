"""Import complete indexed web-reader snapshots, preserving their distinct provenance.

These are potentially cached text extractions, NOT live HTML downloads. Human review remains
mandatory. Raw extracts and corpus stay local; only source metadata and hashes may be published.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from binance_doctalk.ingest import allowed_url
from binance_doctalk.io import digest, read_jsonl, write_jsonl


def extract_snapshot(raw: str, source: dict, metadata: dict) -> dict:
    lines = [(int(n), text) for n, text in re.findall(r'^L(\d+): ?(.*)$', raw, re.M)]
    total = re.search(r'Total lines: (\d+)', raw)
    if not total or [n for n,_ in lines] != list(range(int(total[1]))):
        raise ValueError('Web extract is incomplete or lacks contiguous line provenance')
    header = re.search(r'^.+ \((https://[^\s)]+)\)$', raw, re.M)
    if not header or not allowed_url(header[1]):
        raise ValueError('Web snapshot URL outside official source allowlist')
    if datetime.fromisoformat(metadata['fetched_at']).utcoffset() is None:
        raise ValueError('Snapshot retrieval time requires a timezone')
    includes = {h.casefold() for h in source.get('include_headings',[])}
    excludes = {h.casefold() for h in source.get('exclude_headings',[])}
    title, heading, stack, paragraphs, provenance = None, 'Introduction', [], [], []
    updated, skip_code = None, False
    for line_no, original in lines:
        if original.startswith("    "):
            continue
        line = re.sub(r'【\d+†([^】]+)】', lambda m: m[1].split('†')[0], original).strip()
        if line.startswith('# '):
            if title: raise ValueError('Multiple article titles in source snapshot')
            title = line[2:].strip(); continue
        if not title: continue
        if line.startswith('Updated '):
            updated = line.removeprefix('Updated '); continue
        if line in {'Share Posts','Related Glossaries','Related Articles','Register an account'}:
            break
        h = re.match(r'^(#{2,6}) (.+)$', line)
        if h:
            level, heading = len(h[1]), h[2].strip()
            if heading.casefold() in {'further reading','disclaimer','related articles'}: break
            stack = [(n,t) for n,t in stack if n < level] + [(level,heading.casefold())]
            skip_code = False
            continue
        if line in {'Code Snippet','[Select][Button: Copy]'} or line.startswith('Example: Placing'):
            skip_code = True
        ancestors = {t for _,t in stack}
        if (skip_code or ancestors & excludes or (includes and not ancestors & includes)
                or not line or original.startswith('    ') or line.startswith('Image:')
                or line.startswith('Learn more:') or len(line) < 20):
            continue
        if original.startswith("    "):
            continue
        line = re.sub(r'^\*\s+', '', line)
        pid = f'p{len(paragraphs)+1:03}'
        paragraphs.append({'paragraph_id':pid,'heading':heading,'text':line})
        provenance.append({'paragraph_id':pid,'snapshot_line':line_no})
    if not title or not paragraphs:
        raise ValueError('No usable in-scope paragraphs extracted')
    return {'doc_id':source['doc_id'],'title':title,'url':header[1],'original_url':source['url'],
            'language':'en','kind':'official','fetched_at':metadata['fetched_at'],
            'source_updated_at':updated,'extractor_version':'web-indexed-text-v1',
            'paragraphs':paragraphs,'content_hash':digest(paragraphs),'review_status':'pending_human_review',
            'collection_method':'web_tool_indexed_extract','possibly_cached':True,
            'timestamp_semantics':'fetched_at is snapshot retrieval time; exact upstream crawl time unknown',
            'source_extract_hash':hashlib.sha256(raw.encode()).hexdigest(),
            'paragraph_provenance':provenance,'include_headings':source.get('include_headings',[]),
            'exclude_headings':source.get('exclude_headings',[])}


def import_extracts(folder: Path, manifest: Path, output: Path):
    if output.exists(): raise ValueError('Corpus exists; choose a new version')
    documents = []
    for source in read_jsonl(manifest):
        raw = (folder/(source['doc_id']+'.txt')).read_text()
        meta = json.loads((folder/(source['doc_id']+'.metadata.json')).read_text())
        documents.append(extract_snapshot(raw, source, meta))
    write_jsonl(output, documents)
    return documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/manifest.v1.jsonl')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    docs = import_extracts(args.folder,args.manifest,args.output)
    print({'documents':len(docs),'paragraphs':sum(len(d['paragraphs']) for d in docs),
           'review_status':'pending_human_review','possibly_cached':True})
