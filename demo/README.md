# Recorded demonstrations

## Actual local-model GPU inference

- [30-second GPU walkthrough](gpu-walkthrough.webm): rendered presentation of real saved Qwen
  outputs, not a browser screen recording; 1280×720 VP8 WebM, no audio.
- [All six outputs](gpu-recorded-preview.html): standalone read-only baseline/RAG comparison,
  with inspectable original-note citations. Open directly in a browser. No live inference.
- [Full recorded JSON](gpu-recorded-results.json): actual unmodified final run payload; no outputs dropped.
- `gpu-preview.png`: rendered introduction frame, not a screenshot.

JSON/schema checks passed. Answer correctness, refusal classification and semantic citation support
remain ungraded; known imperfections are discussed in `docs/error_analysis.md`. Official corpus
review and held-out metrics remain pending.

Rebuild these presentation artifacts from saved output (Pillow required):

```bash
python scripts/build_gpu_demo.py artifacts/local/kaggle/gpu-final-output/gpu-smoke.json
GST_REGISTRY=/tmp/doctalk-gst-registry gst-launch-1.0 -q \
  filesrc location=artifacts/local/gpu-walkthrough.avi ! avidemux ! jpegdec ! \
  videoconvert ! vp8enc deadline=1 ! webmmux ! filesink location=demo/gpu-walkthrough.webm
```

## Original deterministic offline preview

- **[Recorded preview](recorded-preview.html):** open directly in a browser, choose one of three
  captured examples, then inspect the original note. This is a replay, not live inference.
- **[27-second WebM walkthrough](offline-walkthrough.webm):** a presentation of three real CLI outputs,
  covering matching notes, a false-premise example and an out-of-scope question. No audio.
- `recorded_outputs.json`: actual outputs captured by `scripts/build_demo.py`.
- `cli-preview.png`: rendered presentation of a captured CLI output; **not a browser screenshot**.

The underlying live offline pipeline runs with `python3 scripts/doctalk.py demo --question "..."`.
These examples use original synthetic notes and a deterministic lexical retrieval/extraction path.
They do not demonstrate LLM reasoning, Binance source quality, or a measured citation benchmark.

To refresh the replay and presentation video, install Pillow in a separate presentation environment:

```bash
python3 -m pip install 'Pillow>=10,<13'
python3 scripts/build_demo.py
GST_REGISTRY=/tmp/doctalk-gst-registry gst-launch-1.0 -q \
  filesrc location=artifacts/local/offline-walkthrough.avi ! avidemux ! jpegdec ! \
  videoconvert ! vp8enc deadline=1 ! webmmux ! filesink location=demo/offline-walkthrough.webm
```

GStreamer is an optional system tool for presentation export, not an application dependency.
Record a new official-source application demo after completing the real study; label its conditions
and distinguish it from these original-note outputs.
