# Offline demonstration artifacts

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
