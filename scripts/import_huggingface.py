"""Import pinned HF Parquet into a private, unapproved source/development candidate pool."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from hashlib import sha256
from html import escape
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from binance_doctalk.benchmark import scope_policy
from binance_doctalk.ingest import allowed_url, load_documents
from binance_doctalk.io import digest, file_hash, read_jsonl, utc_now, write_json, write_jsonl
from binance_doctalk.retrieval import terms


def academy_key(url: str) -> str | None:
    """URL join key only; this is not an observed redirect or proof of text identity."""
    if not allowed_url(url):
        return None
    path = urlsplit(url).path.rstrip('/')
    if path.startswith('/en/academy/'):
        path = path.replace('/en/academy/', '/en/', 1)
    if not path.startswith(('/en/articles/', '/en/glossary/')):
        return None
    return 'binance-academy' + path


def validate_spec(spec):
    if (not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', spec['repo_id'])
            or not re.fullmatch(r'[a-f0-9]{40}', spec['revision'])
            or not re.fullmatch(r'[a-f0-9]{64}', spec['sha256'])
            or not re.fullmatch(r'data/train-\d{5}-of-\d{5}\.parquet', spec['file'])
            or not isinstance(spec['bytes'], int) or not 0 < spec['bytes'] <= 50_000_000):
        raise ValueError('Use a pinned dataset revision, safe Parquet path, SHA-256 and bounded size')


def fetch(spec: dict, cache: Path) -> tuple[Path, dict]:
    validate_spec(spec)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (spec['sha256'] + '.parquet')
    sidecar = cache / (spec['sha256'] + '.json')
    url = f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/{spec['revision']}/{spec['file']}"
    if target.exists() or sidecar.exists():
        if not target.is_file() or not sidecar.is_file():
            raise ValueError('Incomplete download cache; preserve it and choose another cache directory')
        metadata = json.loads(sidecar.read_text())
        if (file_hash(target) != spec['sha256'] or target.stat().st_size != spec['bytes']
                or metadata.get('download_url') != url or metadata.get('sha256') != spec['sha256']):
            raise ValueError('Cached artifact or provenance mismatch; no automatic overwrite')
        from datetime import datetime
        if datetime.fromisoformat(metadata['retrieved_at']).utcoffset() is None:
            raise ValueError('Cached retrieval time needs a timezone')
        return target, metadata
    request = Request(url, headers={'User-Agent': 'BinanceDocTalk/0.1 (dataset provenance import)'})
    with urlopen(request, timeout=45) as response:
        payload = response.read(spec['bytes'] + 1)
    if len(payload) != spec['bytes'] or sha256(payload).hexdigest() != spec['sha256']:
        raise ValueError('Downloaded file size/SHA-256 does not match the pinned dataset artifact')
    metadata = {'repo_id': spec['repo_id'], 'revision': spec['revision'], 'file': spec['file'],
                'download_url': url, 'sha256': spec['sha256'], 'bytes': len(payload),
                'retrieved_at': utc_now(), 'original_page_retrieved_at': None}
    target.write_bytes(payload)
    write_json(sidecar, metadata)
    return target, metadata


def parquet_rows(path: Path, spec: dict) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError('Install the optional importer: python -m pip install -r requirements-datasets.txt') from exc
    table = pq.read_table(path)
    if table.num_rows != spec['rows'] or set(table.column_names) != set(spec['columns']):
        raise ValueError('Parquet row count/schema differs from the pinned dataset contract')
    return table.to_pylist()


def split_markdown(markdown: str, title: str):
    """Keep literal text and character provenance; no summaries or guessed page dates."""
    paragraphs, provenance = [], []
    heading = title
    for match in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', markdown):
        text = match.group().rstrip()
        if re.fullmatch(r'#{1,6} [^\n]+', text):
            heading = text.lstrip('#').strip()
            continue
        if not text:
            continue
        pid = f'p{len(paragraphs)+1:03}'
        paragraphs.append({'paragraph_id': pid, 'heading': heading, 'text': text})
        provenance.append({'paragraph_id': pid, 'markdown_start': match.start(),
                           'markdown_end': match.start()+len(text)})
    if not paragraphs:
        raise ValueError('Selected mirror has no usable text')
    return paragraphs, provenance


def overlap_candidates(question: str, benchmark: list[dict]) -> list[dict]:
    ranked = []
    a = set(terms(question))
    for row in benchmark:
        b = set(terms(row['question']))
        jaccard = len(a & b)/len(a | b) if a | b else 0
        sequence = SequenceMatcher(None, question.casefold(), row['question'].casefold()).ratio()
        ranked.append({'question_id': row['question_id'], 'split': row['split'],
                       'question': row['question'], 'family_id': row['family_id'],
                       'lexical_score': max(jaccard, sequence), 'human_decision': None})
    return sorted(ranked, key=lambda r: (-r['lexical_score'], r['question_id']))[:3]


def qa_flags(row: dict, answer_counts: Counter) -> list[str]:
    q, a = row['question'].casefold(), row['answer'].casefold()
    flags = ['scope_and_evidence_unverified']
    if 'what is what is' in q or 'features of what is' in q or '??' in q:
        flags.append('question_template_artifact')
    if row['question_type'] == 'comparison':
        flags.append('verify_comparison_is_actually_answered')
    if row['question_type'] == 'procedural':
        flags.append('procedural_style_requires_scope_check')
    if 'fee' in q:
        flags.append('check_maker_taker_exception_and_exclude_current_fees')
    if any(w in a for w in ['illegal', 'legally', 'regulated market']):
        flags.append('legal_claim_requires_scope_review')
    if any(w in a for w in ['guarantee', 'without downtime', 'executing right away']):
        flags.append('check_absolute_or_execution_certainty_wording')
    if answer_counts[row['answer']] > 1:
        flags.append('upstream_answer_reused_across_questions')
    return flags


def select_rows(corpus: list[dict], qa: list[dict], current: list[dict], lock: dict,
                downloads: dict, benchmark: list[dict], current_documents=None):
    by_key = {}
    for index, row in enumerate(corpus):
        if row['source'] != 'academy.binance.com':
            continue
        key = academy_key(row['url'])
        if not key:
            raise ValueError('A Binance-labelled row has an invalid official source URL')
        if key in by_key:
            raise ValueError('Duplicate normalized Binance URLs; resolve before importing')
        by_key[key] = (index, row)
    existing = defaultdict(list)
    for row in current:
        key = academy_key(row['url'])
        if key:
            existing[key].append(row['doc_id'])
    additions = {academy_key(row['url']): row['reason'] for row in lock['additional_sources']}
    if None in additions or not additions.keys() <= by_key.keys():
        raise ValueError('Supplemental allowlist contains a missing or invalid article')
    keys = sorted((existing.keys() & by_key.keys()) | additions.keys())
    if len(keys) > lock['max_candidate_documents']:
        raise ValueError('Candidate selection exceeds the MVP document limit')
    docs, public_docs, comparisons = [], [], []
    current_by_id = {r['doc_id']: r for r in current_documents or []}
    for key in keys:
        row_index, row = by_key[key]
        paragraphs, provenance = split_markdown(row['markdown'], row['title'])
        content_hash = digest(paragraphs)
        hf_id = 'HF-' + digest(key)[:12]
        origin = {**downloads['corpus'], 'row_index': row_index, 'row_sha256': digest(row),
                  'markdown_sha256': sha256(row['markdown'].encode()).hexdigest()}
        doc = {'doc_id': hf_id, 'title': row['title'], 'url': row['url'], 'source_url': row['url'],
               'original_url': row['url'], 'language': 'en', 'kind': 'official',
               'version': lock['selection_version'], 'fetched_at': origin['retrieved_at'],
               'retrieved_at': origin['retrieved_at'], 'source_updated_at': None,
               'original_page_retrieved_at': None,
               'timestamp_semantics': 'retrieved_at is the HF artifact download time; original page crawl/update times unknown',
               'collection_method': 'huggingface_mirror', 'upstream_provenance': origin,
               'source_authority': 'Official URL attributed by a third-party dataset; correspondence unverified',
               'extractor_version': 'literal-markdown-blocks-v1', 'paragraphs': paragraphs,
               'paragraph_provenance': provenance, 'content_hash': content_hash,
               'content_sha256': content_hash, 'review_status': 'pending_human_review',
               'reviewer': None, 'reviewed_at': None, 'curation_status': 'not_scope_filtered',
               'matches_current_doc_ids': sorted(existing.get(key, [])),
               'selection_reason': additions.get(key, 'URL overlap with the current source manifest; compare snapshot content'),
               'adoption_status': 'external_pool_only',
               'license_declared_by_publisher': lock['license_declared_by_publisher'],
               'rights_review_status': lock['rights_review_status']}
        docs.append(doc)
        public_docs.append({k:v for k,v in doc.items() if k not in {'paragraphs', 'paragraph_provenance'}})
        for current_id in existing.get(key, []):
            previous = current_by_id.get(current_id)
            old_text = '\n\n'.join(p['text'] for p in previous['paragraphs']) if previous else ''
            new_terms, old_terms = set(terms(row['markdown'])), set(terms(old_text))
            comparisons.append({'current_doc_id': current_id, 'hf_doc_id': hf_id,
                'current_content_hash': previous['content_hash'] if previous else None,
                'mirror_content_hash': content_hash,
                'current_retrieved_at': previous.get('fetched_at') if previous else None,
                'mirror_downloaded_at': origin['retrieved_at'],
                'normalized_text_equal': (' '.join(old_text.split()) == ' '.join(row['markdown'].split())) if previous else None,
                'token_jaccard_not_correctness': len(new_terms & old_terms)/len(new_terms | old_terms) if previous and new_terms | old_terms else None,
                'review_status': 'pending_human_review'})
    selected = {academy_key(d['url']): d for d in docs}
    answer_counts = Counter(row['answer'] for row in qa)
    candidates, public_qa = [], []
    seen_questions = set()
    for index, row in enumerate(qa):
        key = academy_key(row['source_url'])
        if key not in selected:
            continue
        doc = selected[key]
        identity = digest([key, row['question']])
        if identity in seen_questions:
            raise ValueError('Duplicate external question; resolve before importing')
        seen_questions.add(identity)
        candidate = {'candidate_id': 'HFDEV-'+identity[:12], 'candidate_format_version': 1,
            'family_id': 'hf-source-'+digest(key)[:12], 'proposed_split': 'dev',
            'question': row['question'], 'upstream_reference_answer': row['answer'],
            'upstream_question_type': row['question_type'], 'upstream_topic': row['topic'],
            'source_url': row['source_url'], 'source_title': row['source_title'],
            'source_doc_id': doc['doc_id'], 'source_content_hash': doc['content_hash'],
            'label': None, 'test_type': None, 'required_points': [], 'evidence_groups': [],
            'upstream_provenance': {**downloads['qa'], 'row_index': index, 'row_sha256': digest(row)},
            'review_flags': qa_flags(row, answer_counts), 'overlap_candidates': overlap_candidates(row['question'], benchmark),
            'overlap_review_status': 'pending_human_review', 'review_status': 'pending_human_review',
            'reviewer': None, 'reviewed_at': None, 'adoption_status': 'external_dev_pool_only',
            'reference_status': 'LLM-generated upstream answer, not verified project gold'}
        candidates.append(candidate)
        public_qa.append({k:v for k,v in candidate.items() if k not in {'question','upstream_reference_answer','overlap_candidates'}} |
                         {'question_sha256': digest(row['question']), 'answer_sha256': digest(row['answer'])})
    all_urls = {academy_key(r['source_url']) for r in qa if academy_key(r['source_url'])}
    report = {'format_version': 1, 'selection_version': lock['selection_version'], 'created_at': utc_now(),
        'downloads': downloads, 'corpus_rows': len(corpus), 'binance_corpus_rows': len(by_key),
        'qa_rows': len(qa), 'binance_qa_rows': sum(academy_key(r['source_url']) is not None for r in qa),
        'binance_qa_unique_urls': len(all_urls), 'selected_source_count': len(docs),
        'current_manifest_matches': sum(len(existing.get(k, [])) for k in keys),
        'current_manifest_unmatched_ids': sorted(r['doc_id'] for r in current if academy_key(r['url']) not in by_key),
        'supplemental_source_count': sum(k not in existing for k in keys),
        'potential_union_url_count_not_active': len(existing.keys() | set(keys)),
        'dev_qa_candidates': len(candidates), 'source_comparisons': comparisons,
        'qa_review_flag_counts': dict(Counter(f for r in candidates for f in r['review_flags'])),
        'human_approved': 0, 'official_quality_measured': False,
        'active_corpus_or_benchmark_modified': False, 'heldout_inference_performed': False,
        'scope_policy': scope_policy(), 'license_declared_by_publisher': lock['license_declared_by_publisher'],
        'rights_review_status': lock['rights_review_status'], 'status': 'external_candidates_pending_review'}
    return docs, candidates, public_docs, public_qa, report


def render_packet(output: Path, docs: list[dict], candidates: list[dict], report: dict, current: list[dict]):
    output.mkdir(parents=True)
    current_by_id = {r['doc_id']: r for r in current}
    by_id = {d['doc_id']: d for d in docs}
    nav = '<nav><a href="index.html">Overview</a> · <a href="sources.html">Mirror sources</a> · <a href="comparisons.html">Current vs mirror</a> · <a href="candidates.html">Development candidates</a></nav>'
    style = 'body{font:16px/1.6 system-ui;max-width:1200px;margin:32px auto;padding:0 20px;color:#17352c}article{border-top:1px solid #bbc;padding:20px 0}pre,p{white-space:pre-wrap;overflow-wrap:anywhere}pre{background:#f3f6f4;padding:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}nav{padding:12px;background:#edf2ef}@media(max-width:750px){.columns{display:block}}'
    def pretty(value):
        return '<pre>'+escape(json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
    def page(name, title, body):
        (output/name).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+escape(title)+'</title><style>'+style+'</style>'+nav+'<h1>'+escape(title)+'</h1>'+body+'</html>')
    page('index.html','Hugging Face candidate review', '<p><strong>External candidates, not an active corpus or gold benchmark.</strong> URLs join documents; they do not establish content identity or answer support. Download dates are mirror retrieval times. Review scope, provenance, extraction, qualifications and rights before adoption. Existing gold, evidence-gap labels and held-out have not been changed.</p>'+pretty(report))
    body = ''
    for d in docs:
        body += '<article id="'+d['doc_id']+'"><h2>'+escape(d['doc_id']+' — '+d['title'])+'</h2><a href="'+escape(d['url'],quote=True)+'">Upstream URL attributed by dataset</a>'+pretty({k:v for k,v in d.items() if k not in {'paragraphs','paragraph_provenance'}})
        for p in d['paragraphs']:
            body += '<section id="'+d['doc_id']+'-'+p['paragraph_id']+'"><h3>'+escape(p['paragraph_id']+' — '+p['heading'])+'</h3><pre>'+escape(p['text'])+'</pre></section>'
        body += '</article>'
    page('sources.html','Full selected mirror text — unreviewed',body)
    body = '<p>Lexical similarity and URL matching are diagnostics, not correctness or freshness judgments. Keep the current and mirror snapshots separate.</p>'
    for comparison in report['source_comparisons']:
        d = by_id[comparison['hf_doc_id']]
        previous = current_by_id.get(comparison['current_doc_id'])
        old = '\n\n'.join(p['text'] for p in previous['paragraphs']) if previous else 'Current corpus not supplied.'
        new = '\n\n'.join(p['text'] for p in d['paragraphs'])
        body += '<article><h2>'+escape(comparison['current_doc_id']+' ↔ '+d['doc_id'])+'</h2>'+pretty(comparison)+'<div class="columns"><section><h3>Current selected snapshot</h3><pre>'+escape(old)+'</pre></section><section><h3>HF mirror snapshot</h3><pre>'+escape(new)+'</pre></section></div></article>'
    page('comparisons.html','Snapshot comparisons',body)
    body = '<p><strong>Development-only candidate pool.</strong> Labels, test types, required facts and evidence are deliberately unassigned. Upstream question_type describes linguistic style and is not our test_type. Correct the question/reference, bind source spans, and clear semantic overlap before proposing a new benchmark version. Do not copy these rows directly into the active evaluation files.</p>'
    for row in candidates:
        body += '<article id="'+row['candidate_id']+'"><h2>'+escape(row['question'])+'</h2><h3>Upstream generated answer — unverified</h3><p>'+escape(row['upstream_reference_answer'])+'</p><a href="sources.html#'+row['source_doc_id']+'">Read the complete source</a>'+pretty({k:v for k,v in row.items() if k not in {'question','upstream_reference_answer'}})+'</article>'
    page('candidates.html','External QA review',body)


def run(lock_path: Path, manifest: Path, current_corpus: Path | None, benchmark_dir: Path,
        cache: Path, output: Path, public_manifest: Path, public_qa: Path, public_report: Path):
    if any(p.exists() for p in [output, public_manifest, public_qa, public_report]):
        raise ValueError('Outputs already exist; preserve prior imports and choose a new version')
    lock = json.loads(lock_path.read_text())
    artifacts, downloads = {}, {}
    for role in ['corpus','qa']:
        path, metadata = fetch(lock['artifacts'][role], cache)
        artifacts[role] = parquet_rows(path, lock['artifacts'][role])
        downloads[role] = metadata
    current = read_jsonl(current_corpus) if current_corpus else []
    benchmark = [r for split in ['smoke','dev','test'] for r in read_jsonl(benchmark_dir/f'{split}.jsonl')]
    docs, candidates, public_docs, qa_manifest, report = select_rows(
        artifacts['corpus'], artifacts['qa'], read_jsonl(manifest), lock, downloads, benchmark, current)
    report['selection_spec_sha256'] = file_hash(lock_path)
    report['current_manifest_sha256'] = file_hash(manifest)
    report['current_corpus_sha256'] = file_hash(current_corpus) if current_corpus else None
    report['benchmark_file_hashes'] = {split: file_hash(benchmark_dir/f'{split}.jsonl') for split in ['smoke','dev','test']}
    write_jsonl(output/'corpus.candidates.jsonl',docs)
    load_documents(output/'corpus.candidates.jsonl',allow_unreviewed=True)
    write_jsonl(output/'qa.candidates.dev.jsonl',candidates)
    write_json(output/'inspection.json',report)
    render_packet(output/'review',docs,candidates,report,current)
    write_jsonl(public_manifest,public_docs)
    write_jsonl(public_qa,qa_manifest)
    write_json(public_report,report)
    return {k:report[k] for k in ['selected_source_count','current_manifest_matches','supplemental_source_count','dev_qa_candidates','human_approved','status']}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock',type=Path,default=ROOT/'data/huggingface.sources.v1.json')
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/manifest.v1.jsonl')
    parser.add_argument('--current-corpus',type=Path,help='Optional local current corpus for side-by-side comparison')
    parser.add_argument('--benchmark',type=Path,default=ROOT/'data/eval/drafts-v2')
    parser.add_argument('--cache',type=Path,default=ROOT/'artifacts/local/huggingface-downloads')
    for name in ['output','public-manifest','public-qa-manifest','public-report']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(run(args.lock,args.manifest,args.current_corpus,args.benchmark,args.cache,
                             args.output,args.public_manifest,args.public_qa_manifest,args.public_report),indent=2))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
