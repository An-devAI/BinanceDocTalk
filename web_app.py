"""Convenient dependency-free app launcher: python3 web_app.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
from binance_doctalk.cli import main

raise SystemExit(main(["serve", *sys.argv[1:]]))
