"""Real local-LLM integration check on ORIGINAL synthetic notes, not a quality study."""
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from binance_doctalk.answering import BASELINE_PROMPT, RAG_PROMPT
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import utc_now, write_json
from binance_doctalk.provider import create_provider
from binance_doctalk.retrieval import build_index
from binance_doctalk.schemas import Config, validate_answer


def main(output):
    config = Config.load(ROOT / 'configs/kaggle-rag.toml')
    rag = create_provider(config)
    baseline = create_provider(replace(config, system='baseline'))
    print('Loading pinned models and warming up GPU...', flush=True)
    started = time.perf_counter()
    rag.warmup()
    setup_seconds = time.perf_counter()-started
    print('GPU warmup complete:', json.dumps(rag.runtime_info()), flush=True)
    docs = load_documents(ROOT / 'data/demo/notes.jsonl')
    index = build_index(docs, config, rag)
    questions = [
        'How is a market order different from a limit order?',
        'A stop-limit order guarantees execution after triggering, right?',
        'What is the exact Spot fee for my account today?',
    ]
    rows = []
    for q in questions:
        # All cases are integration fixtures, not questions from dev or held-out.
        for mode, provider in [('baseline', baseline), ('rag', rag)]:
            begin = time.perf_counter()
            before = dict(provider.usage)
            context = []
            row = {'question': q, 'mode': mode}
            try:
                if mode == 'rag':
                    v = provider.embed([q], purpose='query')[0]
                    hits = index.search(q, config.top_k, v)
                    context = [{'chunk_id': h['chunk'].chunk_id, 'heading': h['chunk'].heading,
                                'text': h['chunk'].text} for h in hits]
                answer = provider.generate(RAG_PROMPT if mode == 'rag' else BASELINE_PROMPT, q, context)
                validate_answer(answer, {c['chunk_id'] for c in context}, mode == 'rag')
                row.update(answer)
            except Exception as exc:
                row.update(status='error', error_type=type(exc).__name__, message=str(exc))
            row.update(latency_ms=(time.perf_counter()-begin)*1000,
                       usage={k: v-before[k] for k,v in provider.usage.items()},
                       raw_output=provider.last_raw_output)
            rows.append(row)
            print(mode, row['status'], round(row['latency_ms']), 'ms', flush=True)
            # Preserve partial results after every output.
            write_json(output, {'status': 'running', 'predictions': rows})
    summary = {'status': 'complete', 'created_at': utc_now(), 'kind': 'synthetic_integration_only',
               'source_sha256': os.getenv('DOCTALK_SOURCE_SHA256'),
               'config': config.to_dict(), 'runtime': rag.runtime_info(),
               'model_setup_seconds': setup_seconds, 'predictions': rows,
               'errors': sum(r['status']=='error' for r in rows), 'api_usd': 0.0,
               'llm_quality_measured': False, 'human_reviewed_gold': False}
    write_json(output, summary)
    print('GPU integration saved:', str(output), flush=True)
    if summary['errors']:
        raise SystemExit('Some outputs failed strict validation; inspect the saved errors')


if __name__ == '__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/local/gpu-smoke.json')
