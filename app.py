"""Optional Streamlit interface: streamlit run app.py."""
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
import streamlit as st

from binance_doctalk.answering import AnswerService
from binance_doctalk.cli import benchmark_status, demo_service
from binance_doctalk.io import load_env
from binance_doctalk.provider import create_provider
from binance_doctalk.retrieval import Index
from binance_doctalk.schemas import Config

load_env()
st.set_page_config(page_title="BinanceDocTalk", page_icon="📚", layout="wide")
st.title("BinanceDocTalk")
st.caption("Independent educational project. Not affiliated with or endorsed by Binance. Not investment advice.")
mode = st.sidebar.selectbox("Mode", ["Offline original-notes preview", "Live RAG", "Live baseline"])
config_path = st.sidebar.text_input("Configuration", "configs/kaggle-rag.toml")
review = benchmark_status()
st.sidebar.info(f"{review['questions']} benchmark cases · {review['human_reviewed']} declared reviewed · load final metrics separately after evaluation")
if mode.startswith("Offline"):
    st.info("Deterministic retrieval of original notes. No LLM or official Binance corpus.")
question = st.chat_input("Ask about crypto terms, Spot, or order types")
if question:
    st.chat_message("user").write(question)
    try:
        demo = mode.startswith("Offline")
        if demo:
            service = demo_service()
        else:
            config = replace(Config.load(Path(config_path)),
                             system="rag" if mode == "Live RAG" else "baseline")
            index = Index.load(Path("data/local/index.v1.json")) if config.system == "rag" else None
            service = AnswerService(index, config, create_provider(config))
        with st.spinner("Retrieving and validating…"):
            result = service.ask(question, demo=demo)
        with st.chat_message("assistant"):
            st.caption("Original-note preview" if demo else result["status"])
            if result["status"] == "error":
                st.error(result["message"])
            elif result["message"]:
                st.info(result["message"])
            for claim in result["claims"]:
                st.write(claim["text"])
                st.caption(" · ".join(claim["evidence_ids"]))
            for hit in result["retrieved"]:
                with st.expander(f"{hit['chunk_id']} — {hit['title']}"):
                    st.text(service.source_text(hit["chunk_id"]))
                    if hit["url"].startswith("https://"):
                        st.link_button("Open source", hit["url"])
            st.caption(f"{result['latency_ms']:.0f} ms · estimated API cost ${result['estimated_usd']:.5f}")
    except (ValueError, RuntimeError, OSError) as exc:
        st.error(str(exc))
