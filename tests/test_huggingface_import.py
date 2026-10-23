"""Pinned mirror imports stay private, unapproved, and separate from evaluation gold."""
from copy import deepcopy
from hashlib import sha256
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from binance_doctalk.ingest import load_documents
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.io import digest, write_jsonl
from binance_doctalk.schemas import Document
from import_huggingface import (academy_key, fetch, parquet_rows, render_packet, run,
                               select_rows, split_markdown, validate_spec)


def fixture_spec(payload=b'UNIT TEST FILE'):
    return dict(repo_id='fixture-owner/fixture-dataset',revision='1'*40,
                file='data/train-00000-of-00001.parquet',sha256=sha256(payload).hexdigest(),
                bytes=len(payload),rows=2,columns=['url','title','markdown','source'])


def fixture_inputs():
    corpus=[dict(url='https://academy.binance.com/en/articles/unit-existing',title='UNIT EXISTING',
                 markdown='# Original heading\n\nUNIT TEST source text.\n\nA second paragraph.',
                 source='academy.binance.com'),
            dict(url='https://academy.binance.com/en/articles/unit-added',title='UNIT ADDED',
                 markdown='UNIT TEST ONLY: an added source, not real crypto knowledge.',source='academy.binance.com'),
            dict(url='https://other.example/article',title='NOT SELECTED',markdown='Not selected.',source='other.example')]
    qa=[dict(question='UNIT question one?',answer='UNIT reference.',source_url=corpus[1]['url'],
             source_title='UNIT ADDED',topic='fixture',question_type='comparison'),
        dict(question='UNIT question two?',answer='UNIT reference.',source_url=corpus[1]['url'],
             source_title='UNIT ADDED',topic='fixture',question_type='informal'),
        dict(question='Not selected?',answer='Not selected.',source_url=corpus[2]['url'],
             source_title='NOT SELECTED',topic='fixture',question_type='factual')]
    manifest=[dict(doc_id='UNIT-OLD',url='https://www.binance.com/en/academy/articles/unit-existing')]
    lock=dict(selection_version='UNIT-TEST-CANDIDATES',max_candidate_documents=30,
              additional_sources=[dict(url=corpus[1]['url'],reason='UNIT TEST ALLOWLIST')],
              license_declared_by_publisher='UNIT TEST ONLY',rights_review_status='not_verified')
    downloads={role:dict(repo_id='fixture-owner/fixture-dataset',revision='1'*40,file='data/train-00000-of-00001.parquet',
                        sha256='2'*64,retrieved_at='2026-01-01T12:00:00+00:00') for role in ['corpus','qa']}
    benchmark=[dict(question_id='UNIT-TEST',family_id='unit-family',split='test',question='UNIT question one?')]
    return corpus,qa,manifest,lock,downloads,benchmark


class HuggingFaceImportTests(unittest.TestCase):
    def test_only_official_https_urls_are_joined_without_inventing_redirects(self):
        old='https://academy.binance.com/en/articles/unit-source?tracking=1#part'
        new='https://www.binance.com/en/academy/articles/unit-source'
        self.assertEqual(academy_key(old),academy_key(new))
        for url in ['http://academy.binance.com/en/articles/unit-source',
                    'https://academy.binance.com.evil.example/en/articles/unit-source',
                    'https://user@academy.binance.com/en/articles/unit-source',
                    'https://academy.binance.com:8443/en/articles/unit-source',
                    'https://academy.binance.com/private/unit-source']:
            self.assertIsNone(academy_key(url))

    def test_artifact_revision_path_and_size_are_bounded(self):
        validate_spec(fixture_spec())
        for field,value in [('revision','main'),('file','../../secret'),('repo_id','../secret'),('bytes',90_000_000)]:
            bad=dict(fixture_spec(),**{field:value})
            with self.assertRaises(ValueError): validate_spec(bad)

    def test_download_pin_and_real_retrieval_time_are_preserved_in_cache(self):
        payload=b'UNIT TEST FILE';spec=fixture_spec(payload)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            with patch('import_huggingface.urlopen',return_value=BytesIO(payload)) as network, \
                 patch('import_huggingface.utc_now',return_value='2026-01-01T12:00:00+00:00'):
                artifact,meta=fetch(spec,path)
                network.assert_called_once()
            with patch('import_huggingface.urlopen',side_effect=AssertionError('Cache must not redownload')):
                cached,again=fetch(spec,path)
            self.assertEqual(again,meta)
            self.assertEqual(cached,artifact)
            self.assertIsNone(meta['original_page_retrieved_at'])
            self.assertEqual(meta['retrieved_at'],'2026-01-01T12:00:00+00:00')
            artifact.write_bytes(b'CORRUPTED')
            with self.assertRaisesRegex(ValueError,'mismatch'):
                fetch(spec,path)
            self.assertEqual(artifact.read_bytes(),b'CORRUPTED')

    def test_wrong_download_hash_never_becomes_a_cached_artifact(self):
        payload=b'UNIT TEST FILE';spec=dict(fixture_spec(payload),sha256='0'*64)
        with tempfile.TemporaryDirectory() as folder:
            with patch('import_huggingface.urlopen',return_value=BytesIO(payload)):
                with self.assertRaisesRegex(ValueError,'does not match'):
                    fetch(spec,Path(folder))
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_parquet_contract_rejects_wrong_row_count_or_columns(self):
        class Table:
            num_rows=66
            column_names=['url','title','markdown','source']
            def to_pylist(self): return []
        parquet=SimpleNamespace(read_table=lambda p:Table())
        arrow=SimpleNamespace(parquet=parquet)
        with patch.dict(sys.modules,{'pyarrow':arrow,'pyarrow.parquet':parquet}):
            with self.assertRaisesRegex(ValueError,'row count/schema'):
                parquet_rows(Path('unused'),dict(fixture_spec(),rows=3487))
            with self.assertRaisesRegex(ValueError,'row count/schema'):
                parquet_rows(Path('unused'),dict(fixture_spec(),rows=66,columns=['url']))

    def test_unicode_paragraph_offsets_recover_literal_source_text(self):
        text='# Title\n\n  Café 🪙 original words.\n\n## Risks\n\nQualifier: may fail.\n'
        paragraphs,provenance=split_markdown(text,'Fallback')
        self.assertEqual(len(paragraphs),2)
        self.assertEqual(paragraphs[1]['heading'],'Risks')
        for p,span in zip(paragraphs,provenance):
            self.assertEqual(text[span['markdown_start']:span['markdown_end']],p['text'])

    def test_selected_qa_is_dev_only_unassigned_and_never_gold(self):
        inputs=fixture_inputs();before=deepcopy(inputs)
        docs,candidates,public_docs,public_qa,report=select_rows(*inputs)
        self.assertEqual(inputs,before)
        self.assertEqual(report['selected_source_count'],2)
        self.assertEqual(report['supplemental_source_count'],1)
        self.assertEqual(report['dev_qa_candidates'],2)
        self.assertFalse(report['active_corpus_or_benchmark_modified'])
        self.assertFalse(report['heldout_inference_performed'])
        self.assertEqual(candidates[0]['family_id'],candidates[1]['family_id'])
        for q in candidates:
            self.assertEqual(q['proposed_split'],'dev')
            self.assertIsNone(q['label']);self.assertIsNone(q['test_type'])
            self.assertFalse(q['evidence_groups'])
            self.assertEqual(q['review_status'],'pending_human_review')
            self.assertEqual(q['overlap_candidates'][0]['split'],'test')
            with self.assertRaisesRegex(ValueError,'cannot be indexed'):
                Document.from_dict(q)
            with self.assertRaisesRegex(ValueError,'not an adopted benchmark'):
                validate_questions([q])
        for d in docs:
            self.assertIsNone(d['source_updated_at'])
            self.assertEqual(d['collection_method'],'huggingface_mirror')
            self.assertEqual(d['url'],d['original_url'])
        self.assertTrue(all('paragraphs' not in d for d in public_docs))
        self.assertTrue(all('question' not in q and 'upstream_reference_answer' not in q for q in public_qa))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'corpus.jsonl';write_jsonl(path,docs)
            load_documents(path,allow_unreviewed=True)
            with self.assertRaisesRegex(ValueError,'human review'):
                load_documents(path)

    def test_missing_duplicate_and_oversized_allowlists_are_rejected(self):
        inputs=list(fixture_inputs())
        bad=deepcopy(inputs);bad[0].append(dict(bad[0][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate normalized'):
            select_rows(*bad)
        bad=deepcopy(inputs);bad[3]['additional_sources'][0]['url']='https://academy.binance.com/en/articles/missing'
        with self.assertRaisesRegex(ValueError,'missing or invalid'):
            select_rows(*bad)
        bad=deepcopy(inputs);bad[3]['max_candidate_documents']=1
        with self.assertRaisesRegex(ValueError,'MVP document limit'):
            select_rows(*bad)
        bad=deepcopy(inputs);bad[0][0]['url']='https://evil.example/en/articles/unit-existing'
        with self.assertRaisesRegex(ValueError,'invalid official source'):
            select_rows(*bad)

    def test_review_html_escapes_untrusted_text_and_local_links_resolve(self):
        inputs=list(fixture_inputs())
        inputs[1][0]['question']='<script>alert(1)</script>'
        inputs[0][1]['markdown']='<img src=x onerror=alert(1)>'
        docs,candidates,_,_,report=select_rows(*inputs)
        class Links(HTMLParser):
            def __init__(self): super().__init__();self.ids=set();self.links=[]
            def handle_starttag(self,tag,attrs):
                a=dict(attrs)
                if 'id' in a:self.ids.add(a['id'])
                if tag=='a' and 'href' in a:self.links.append(a['href'])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'review';render_packet(root,docs,candidates,report,[])
            pages={}
            for p in root.glob('*.html'):
                text=p.read_text();self.assertNotIn('<script>',text);self.assertNotIn('<img ',text)
                parser=Links();parser.feed(text);pages[p.name]=parser
            for parser in pages.values():
                for link in parser.links:
                    if link.startswith('https://'):continue
                    name,_,anchor=link.partition('#')
                    self.assertIn(name,pages)
                    if anchor:self.assertIn(anchor,pages[name].ids)

    def test_existing_output_is_not_overwritten_or_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            home=Path(folder);output=home/'existing';output.mkdir()
            with patch('import_huggingface.fetch',side_effect=AssertionError('No network expected')):
                with self.assertRaisesRegex(ValueError,'already exist'):
                    run(home/'lock',home/'manifest',None,home/'benchmark',home/'cache',output,
                        home/'manifest-out',home/'qa-out',home/'report-out')


if __name__=='__main__':unittest.main()
