"""Import legitimately obtained HTML snapshots with explicitly recorded collection times.

Input folder: D01.html + D01.metadata.json, etc. Sidecar requires fetched_at (actual collection
time, ISO 8601 with timezone) and url (the actual final Binance document URL). No browser evasion.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from binance_doctalk.ingest import allowed_url, extract_html
from binance_doctalk.io import read_jsonl, write_jsonl


def import_snapshots(folder: Path, manifest: Path, output: Path):
    if output.exists():
        raise ValueError('Corpus exists; choose a new version')
    documents = []
    for source in read_jsonl(manifest):
        html_path = folder/(source['doc_id']+'.html')
        if not html_path.exists():
            continue
        metadata = json.loads(html_path.with_suffix('.metadata.json').read_text())
        if not allowed_url(metadata['url']):
            raise ValueError('Snapshot URL outside official source allowlist')
        if datetime.fromisoformat(metadata['fetched_at']).utcoffset() is None:
            raise ValueError('Actual fetched_at needs a timezone')
        row = extract_html(html_path.read_text(), source, metadata['fetched_at'])
        row.update(url=metadata['url'], collection_method='manual_html_import',
                   review_status='pending_human_review')
        documents.append(row)
    if not documents:
        raise ValueError('No named HTML snapshots found; no corpus created')
    write_jsonl(output, documents)
    return {'imported': len(documents), 'review_status': 'pending_human_review'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/manifest.v1.jsonl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(import_snapshots(args.folder, args.manifest, args.output))
