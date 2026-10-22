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
from import_web_extracts import extract_snapshot
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

    def test_indexed_snapshot_requires_complete_provenance_and_retains_cache_status(self):
        raw = ("Original test fixture (https://www.binance.com/en/academy/articles/unit-test)\n"
               "Total lines: 8\nL0: # Test title\nL1: Updated Jan 1, 2026\nL2: ## Keep\n"
               "L3: A synthetic paragraph with 【1†inline linked text】 for testing.\n"
               "L4:     # A code comment is not a second article title\n"
               "L5: ## Further Reading\nL6: Do not include this unrelated paragraph.\nL7: \n")
        source = {'doc_id':'TEST','url':'https://www.binance.com/en/academy/articles/unit-test'}
        meta = {'fetched_at':'2026-01-01T12:00:00+00:00'}
        row = extract_snapshot(raw, source, meta)
        self.assertEqual(len(row['paragraphs']), 1)
        self.assertIn('inline linked text', row['paragraphs'][0]['text'])
        self.assertEqual(row['paragraph_provenance'][0]['snapshot_line'], 3)
        self.assertEqual(row['review_status'], 'pending_human_review')
        self.assertTrue(row['possibly_cached'])
        self.assertIn('upstream crawl time unknown', row['timestamp_semantics'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            extract_snapshot(raw.replace('L7: \n',''), source, meta)

    def test_snapshot_scope_includes_descendants_but_excludes_procedural_code(self):
        raw = ("Original test fixture (https://www.binance.com/en/academy/articles/unit-test)\n"
               "Total lines: 9\nL0: # Test title\nL1: ## Allowed\nL2: ### Detail\n"
               "L3: A useful synthetic explanation of an order parameter.\n"
               "L4: Code Snippet\nL5: This procedural content is excluded from the corpus.\n"
               "L6: ## Other\nL7: Another paragraph outside the selected parent section.\nL8: Share Posts\n")
        source = {'doc_id':'TEST','url':'https://www.binance.com/en/academy/articles/unit-test',
                  'include_headings':['Allowed']}
        row = extract_snapshot(raw, source, {'fetched_at':'2026-01-01T12:00:00+00:00'})
        self.assertEqual(len(row['paragraphs']), 1)
        self.assertEqual(row['paragraphs'][0]['heading'],'Detail')

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
