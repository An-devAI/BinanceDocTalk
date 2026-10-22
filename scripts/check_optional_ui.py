"""Optional Streamlit runtime checks, not browser visual QA or LLM evaluation."""
from pathlib import Path
import importlib.metadata
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from streamlit.testing.v1 import AppTest
from binance_doctalk.chunking import token_offsets
from binance_doctalk.io import utc_now, write_json


def main(output: Path):
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()
    assert not list(app.exception), [e.message for e in app.exception]
    app.chat_input[0].set_value('Why can a stop-limit order remain unfilled?').run()
    assert not list(app.exception), [e.message for e in app.exception]
    assert 'does not guarantee execution' in '\n'.join(m.value for m in app.markdown)
    assert 'DEMO02:c0001' in '\n'.join(c.value for c in app.caption)
    app.chat_input[0].set_value('What is the current Spot fee for my account?').run()
    assert not list(app.exception), [e.message for e in app.exception]
    assert 'cannot provide current/account data' in '\n'.join(m.value for m in app.info)
    text='Crypto café 🚀 market liquidity.'
    offsets=token_offsets(text,'cl100k_base')
    assert offsets and offsets[0][0]==0 and offsets[-1][1]==len(text)
    assert all(0<=a<b<=len(text) for a,b in offsets)
    summary={'created_at':utc_now(),'python':sys.version.split()[0],
             'streamlit':importlib.metadata.version('streamlit'),
             'tiktoken':importlib.metadata.version('tiktoken'),
             'streamlit_initial_render':True,'offline_answer_with_citation':True,
             'out_of_scope_abstention':True,'actual_unicode_token_offsets':True,
             'method':'Streamlit AppTest and tokenizer execution; no browser visual QA',
             'llm_called':False}
    write_json(output,summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/local/optional-ui-check.json')
