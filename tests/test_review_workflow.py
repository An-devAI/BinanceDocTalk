"""Review tooling preserves pending status and exact source provenance."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from import_sources import import_snapshots
from prepare_review import prepare
from binance_doctalk.ingest import collect
from binance_doctalk.io import read_jsonl, write_jsonl
from test_pipeline import fixture_doc


class ReviewWorkflowTests(unittest.TestCase):
    def test_empty_corpus_packet_is_explicitly_blocked_and_never_approves_drafts(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'packet'
            summary = prepare(output, None, ROOT/'data/eval/drafts', ROOT/'data/manifest.v1.jsonl')
            self.assertEqual(summary['questions'], 80)
            self.assertEqual(summary['status'], 'blocked_missing_corpus')
            for split in ['smoke','dev','test']:
                for row in read_jsonl(output/f'{split}.suggestions.jsonl'):
                    self.assertFalse(row['human_approved'])
                    self.assertNotEqual(row['draft']['review_status'], 'human_reviewed')
                    self.assertTrue(all(not s['candidates'] for s in row['suggestions']))
            with self.assertRaisesRegex(ValueError, 'exists'):
                prepare(output, None, ROOT/'data/eval/drafts', ROOT/'data/manifest.v1.jsonl')

    def test_candidate_span_matches_exact_local_paragraph_and_html_is_escaped(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            corpus = home/'corpus.jsonl'
            doc = fixture_doc('official')
            write_jsonl(corpus, [asdict(doc)])
            questions = home/'questions';questions.mkdir()
            row = dict(read_jsonl(ROOT/'data/eval/drafts/smoke.jsonl')[0])
            row['question'] = '<script>alert(1)</script> alpha beta'
            row['required_points'] = ['alpha beta']
            for split in ['smoke','dev','test']:write_jsonl(questions/f'{split}.jsonl', [row])
            prepare(home/'packet', corpus, questions, ROOT/'data/manifest.v1.jsonl')
            data = read_jsonl(home/'packet/smoke.suggestions.jsonl')[0]
            candidate = data['suggestions'][0]['candidates'][0]
            self.assertEqual(candidate['text'], doc.paragraphs[0].text)
            self.assertEqual(candidate['end'], len(candidate['text']))
            self.assertEqual(candidate['content_hash'], doc.content_hash)
            self.assertNotIn('<script>',(home/'packet/smoke.html').read_text())

    def test_html_import_requires_actual_timestamp_and_stays_unreviewed(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            snapshots = home/'snapshots';snapshots.mkdir()
            (snapshots/'D01.html').write_text('<h1>Original test fixture</h1><p>A synthetic paragraph with enough text for extraction.</p>')
            sidecar = snapshots/'D01.metadata.json'
            sidecar.write_text(json.dumps({'url':'https://www.binance.com/en/academy/articles/unit-test',
                                          'fetched_at':'2026-01-01T12:00:00'}))
            with self.assertRaisesRegex(ValueError,'timezone'):
                import_snapshots(snapshots,ROOT/'data/manifest.v1.jsonl',home/'corpus.jsonl')
            sidecar.write_text(json.dumps({'url':'https://www.binance.com/en/academy/articles/unit-test',
                                          'fetched_at':'2026-01-01T12:00:00+00:00'}))
            result = import_snapshots(snapshots,ROOT/'data/manifest.v1.jsonl',home/'corpus.jsonl')
            self.assertEqual(result['imported'],1)
            doc = read_jsonl(home/'corpus.jsonl')[0]
            self.assertEqual(doc['review_status'],'pending_human_review')
            self.assertEqual(doc['fetched_at'],'2026-01-01T12:00:00+00:00')
            self.assertEqual(doc['collection_method'],'manual_html_import')

    def test_http_202_does_not_create_an_official_corpus(self):
        class Response:
            status = 202
            url = 'https://www.binance.com/robots.txt'
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): return b'User-agent: *\nAllow: /\n'
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            manifest = home/'manifest.jsonl'
            write_jsonl(manifest,[read_jsonl(ROOT/'data/manifest.v1.jsonl')[0]])
            with (patch('binance_doctalk.ingest.build_opener') as opener,
                  patch('binance_doctalk.ingest.time.sleep')):
                opener.return_value.open.return_value = Response()
                result = collect(manifest, home/'corpus.jsonl', True)
            self.assertEqual(result['collected'],0)
            self.assertIn('HTTP 202',result['failures'][0]['message'])
            self.assertFalse((home/'corpus.jsonl').exists())
