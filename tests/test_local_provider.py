"""CPU-safe contract tests; no model downloads or inference claims."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from binance_doctalk.chunking import token_offsets, chunk_document
from binance_doctalk.local_provider import (LocalProvider, QUERY_PREFIX, gpu_preflight,
                                          generation_runtime)
from binance_doctalk.provider import create_provider, ProviderError
from binance_doctalk.schemas import Config, validate_answer
from test_pipeline import fixture_doc


def local_config():
    return Config.load(Path(__file__).resolve().parents[1] / 'configs/kaggle-rag.toml')


class LocalProviderTests(unittest.TestCase):
    def test_factory_selects_local_without_key_or_loading_weights(self):
        with patch('binance_doctalk.local_provider.generation_runtime') as loader:
            provider = create_provider(local_config())
            self.assertIsInstance(provider, LocalProvider)
            self.assertEqual(provider.ledger.read()['estimated_usd'], 0)
            self.assertIsNone(provider.runtime_info()['compute_usd'])
            loader.assert_not_called()

    def test_config_requires_pinned_matching_models(self):
        with self.assertRaisesRegex(ValueError, 'pinned'):
            replace(local_config(), model_revision='main')
        with self.assertRaisesRegex(ValueError, 'matching tokenizer'):
            replace(local_config(), tokenizer='regex_word_v1')
        with self.assertRaisesRegex(ProviderError, 'revision'):
            LocalProvider(replace(local_config(), embedding_revision='0'*40))

    def test_missing_gpu_and_low_vram_have_explicit_failures(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(ProviderError, 'CUDA GPU unavailable'):
            gpu_preflight(torch)
        torch.cuda = SimpleNamespace(is_available=lambda: True,
            get_device_properties=lambda _: SimpleNamespace(name='synthetic'),
            mem_get_info=lambda _: (1_000_000_000, 16_000_000_000))
        with self.assertRaisesRegex(ProviderError, 'free VRAM'):
            gpu_preflight(torch)

    def test_query_prefix_normalization_and_no_silent_embedding_truncation(self):
        inputs_seen = []
        encoder = SimpleNamespace(max_seq_length=512,
            tokenizer=lambda text, **kwargs: {'input_ids': list(range(len(text.split())+2))})
        def encode(inputs, **kwargs):
            inputs_seen.append((inputs, kwargs))
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in inputs])
        encoder.encode = encode
        with patch('binance_doctalk.local_provider.embedding_runtime', return_value=encoder):
            p = LocalProvider(local_config())
            self.assertEqual(p.embed(['market'], purpose='query'), [[1., 0.]])
            p.embed(['market'])
            self.assertEqual(inputs_seen[0][0], [QUERY_PREFIX+'market'])
            self.assertEqual(inputs_seen[1][0], ['market'])
            self.assertTrue(inputs_seen[0][1]['normalize_embeddings'])
            with self.assertRaisesRegex(ProviderError, 'truncation'):
                p.embed(['word '*600])

    def test_fast_tokenizer_offsets_preserve_unicode_evidence(self):
        text = 'Crypto café is liquid.'
        def tokenizer(value, **kwargs):
            import re
            return {'offset_mapping': [m.span() for m in re.finditer(r'\S+', value)]}
        with patch('binance_doctalk.chunking.bge_tokenizer', return_value=tokenizer):
            offsets = token_offsets(text, 'bge_small_v1_5')
            self.assertEqual(text[slice(*offsets[1])], 'café')
            doc = fixture_doc()
            chunks = chunk_document(doc, 6, 2, 'bge_small_v1_5')
            for chunk in chunks:
                self.assertTrue(chunk.spans)
                for span in chunk.spans:
                    para = next(p for p in doc.paragraphs if p.paragraph_id == span['paragraph_id'])
                    self.assertIn(para.text[span['start']:span['end']], chunk.text)

    def fake_runtime(self, text, input_count=32):
        class Tokens(list):
            @property
            def shape(self): return (1, input_count)
        class Inputs(dict):
            def to(self, device): return self
        tokenizer = SimpleNamespace(
            apply_chat_template=lambda *a, **k: Inputs(input_ids=Tokens()),
            eos_token_id=0, decode=lambda *a, **k: text)
        from contextlib import nullcontext
        torch = SimpleNamespace(manual_seed=lambda _: None, inference_mode=nullcontext,
            cuda=SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda _: 123))
        model = SimpleNamespace(generate=lambda **kwargs: [[1]*input_count+[2, 0]])
        return torch, tokenizer, model, {}

    def test_invalid_json_is_error_without_repair_and_usage_is_retained(self):
        with patch('binance_doctalk.local_provider.generation_runtime',
                   return_value=self.fake_runtime('```json\n{}\n```')):
            p = LocalProvider(local_config())
            with self.assertRaisesRegex(ProviderError, 'not repaired'):
                p.generate('instructions', 'question', [])
            self.assertEqual(p.usage['output_tokens'], 2)
            self.assertEqual(p.retry_count, 0)

    def test_oversized_prompt_is_rejected_before_generation(self):
        with patch('binance_doctalk.local_provider.generation_runtime',
                   return_value=self.fake_runtime('{}', 5000)):
            with self.assertRaisesRegex(ProviderError, 'not truncated'):
                LocalProvider(local_config()).generate('instructions', 'question', [])

    def test_local_json_still_passes_shared_citation_validator(self):
        bad = {'status':'answered','reason':None,'message':'','claims':[
            {'text':'A claim','evidence_ids':['INVENTED']}]}
        with patch('binance_doctalk.local_provider.generation_runtime',
                   return_value=self.fake_runtime(json.dumps(bad))):
            answer = LocalProvider(local_config()).generate('instructions','question',[])
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                validate_answer(answer, {'REAL'}, True)

    def test_generator_loader_is_process_cached_for_paired_providers(self):
        self.assertEqual(generation_runtime.cache_parameters()['maxsize'], 1)
        a,b = LocalProvider(local_config()),LocalProvider(replace(local_config(), system='baseline'))
        with patch('binance_doctalk.local_provider.generation_runtime',
                   return_value=self.fake_runtime('{}')) as loader:
            a.generate('rules','a',[]); b.generate('rules','b',[])
            self.assertEqual(loader.call_args_list[0], loader.call_args_list[1])
