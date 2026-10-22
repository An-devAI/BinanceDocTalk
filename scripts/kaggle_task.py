"""Prepare, push, monitor and download a private CPU or GPU Kaggle smoke notebook."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

from package_source import ROOT, source_archive


def environment() -> dict:
    env = dict(os.environ)
    path = ROOT / ".env.kaggle"
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                if key in {"KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"}:
                    env.setdefault(key, value.strip().strip("\"'"))
    # Reuse existing OAuth, without asking for another token or copying a refresh token.
    oauth = Path.home() / ".kaggle/credentials.json"
    if oauth.is_file():
        values = json.loads(oauth.read_text())
        if values.get("username"):
            env.setdefault("KAGGLE_USERNAME", values["username"])
        expiration = values.get("access_token_expiration")
        if expiration and values.get("access_token"):
            expires_at = datetime.fromisoformat(expiration)
            if expires_at.utcoffset() is not None and expires_at > datetime.now(timezone.utc):
                env.setdefault("KAGGLE_API_TOKEN", values["access_token"])
    env.setdefault("KAGGLE_CONFIG_DIR", str(ROOT / "artifacts/local/kaggle/config"))
    return env


def cli_command() -> list[str] | None:
    local = ROOT / ".venv-kaggle/bin/python"
    if local.is_file():
        # A partially created venv is not a successfully installed CLI.
        probe = subprocess.run([str(local), "-c", "import importlib.metadata; importlib.metadata.version('kaggle')"],
                               capture_output=True)
        if probe.returncode == 0:
            return [str(local), "-m", "kaggle"]
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    spec = importlib.util.find_spec("kaggle")
    if spec is not None and spec.origin is not None:
        return [sys.executable, "-m", "kaggle"]
    return None


def cli_environment(env: dict) -> dict:
    env = dict(env)
    # Username alone makes CLI 2.2.4 attempt incomplete legacy-key authentication,
    # which prevents OAuth refresh. Keep it for notebook ownership, not CLI auth.
    if not env.get("KAGGLE_KEY"):
        env.pop("KAGGLE_USERNAME", None)
    return env


def doctor(check_auth=False) -> dict:
    env = environment()
    command = cli_command()
    config = Path(env["KAGGLE_CONFIG_DIR"])
    auth_present = bool(env.get("KAGGLE_API_TOKEN") or
                        (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")) or
                        (config/"access_token").is_file() or (config/"kaggle.json").is_file())
    oauth_present = (Path.home()/".kaggle/credentials.json").is_file()
    if oauth_present:
        oauth_values = json.loads((Path.home()/".kaggle/credentials.json").read_text())
        # The official SDK refreshes expired access tokens using its saved OAuth credential.
        auth_present = auth_present or bool(oauth_values.get("refresh_token"))
    try:
        socket.getaddrinfo("www.kaggle.com", 443)
        dns = True
    except OSError:
        dns = False
    result = {"cli_available": command is not None, "credentials_present": auth_present,
              "username_present": bool(env.get("KAGGLE_USERNAME")), "dns_resolves": dns,
              "existing_oauth_file_present": oauth_present,
              "authenticated": None, "remote_execution": "not_started"}
    if check_auth and command and auth_present and dns:
        config.mkdir(parents=True, exist_ok=True)
        probe = subprocess.run([*command, "kernels", "list", "--mine", "--page-size", "1"],
                               env=cli_environment(env), capture_output=True, text=True, timeout=60)
        result["authenticated"] = probe.returncode == 0
        # Never echo CLI errors or response bodies that might contain credential diagnostics.
        result["auth_exit_code"] = probe.returncode
    return result


def cell(text: str, code=False) -> dict:
    value = {"cell_type": "code" if code else "markdown", "metadata": {},
             "source": text.splitlines(keepends=True), "id": hashlib.sha256(text.encode()).hexdigest()[:12]}
    if code:
        value.update(execution_count=None, outputs=[])
    return value


def prepare(owner: str, output: Path) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", owner):
        raise ValueError("Use a Kaggle username slug, not an email or token")
    archive = source_archive(include_media=False)
    sha = hashlib.sha256(archive).hexdigest()
    encoded = base64.b64encode(archive).decode()
    bootstrap = f'''# Embedded allowlisted source only: no .env, tokens, local corpus or indexes.
import base64, hashlib, io, json, os, subprocess, sys, zipfile
from pathlib import Path
PAYLOAD = "{encoded}"
EXPECTED_SHA256 = "{sha}"
raw = base64.b64decode(PAYLOAD)
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256
work = Path(os.environ.get("DOCTALK_WORK_ROOT", "/kaggle/working"))
project = work / "binance-doctalk"
if project.exists():
    raise RuntimeError("Project directory already exists; preserve it and use a fresh session")
project.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(raw)) as archive:
    for item in archive.infolist():
        (project / item.filename).resolve().relative_to(project.resolve())
    archive.extractall(project)
os.chdir(project)
print("Source verified:", EXPECTED_SHA256)
print("Runtime Python:", sys.version.split()[0])
'''
    tests = '''# Core tests require no downloads, model key, GPU or paid calls.
result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                        capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
(work / "core-tests.txt").write_text(result.stdout + result.stderr)
assert result.returncode == 0, "Core tests failed; do not report quality results"
audit = subprocess.run([sys.executable, "scripts/doctalk.py", "audit-benchmark"],
                       capture_output=True, text=True, check=True)
print(audit.stdout)
(work / "benchmark-audit.json").write_text(audit.stdout)
'''
    demo = '''# Real execution of the original-notes preview, not replayed or LLM-generated output.
questions = [
    "How is a market order different from a limit order?",
    "A stop-limit order guarantees execution after triggering, right?",
    "What is the exact Spot fee for my account today?",
]
outputs = []
for question in questions:
    process = subprocess.run([sys.executable, "scripts/doctalk.py", "demo", "--question", question],
                             capture_output=True, text=True, check=True)
    answer = json.loads(process.stdout)
    outputs.append(answer)
    print("\\nQUESTION:", question, "\\nSTATUS:", answer["status"])
    for claim in answer["claims"]:
        print(claim["text"], claim["evidence_ids"])
    if answer["message"]:
        print(answer["message"])
(work / "offline-demo.json").write_text(json.dumps(outputs, indent=2))
print("\\nOfficial corpus: not collected; human gold review pending; LLM quality NOT MEASURED.")
'''
    probe = '''# Optional owner-enabled connectivity probe. This is NOT a held-out evaluation.
RUN_PAID_OPENAI_PROBE = False
if RUN_PAID_OPENAI_PROBE:
    from kaggle_secrets import UserSecretsClient
    # Add OPENAI_API_KEY in Kaggle Add-ons -> Secrets and grant this notebook access.
    os.environ["OPENAI_API_KEY"] = UserSecretsClient().get_secret("OPENAI_API_KEY")
    process = subprocess.run([sys.executable, "scripts/doctalk.py", "ask",
        "What is a market order?", "--config", "configs/baseline.toml"],
        capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError("Live probe failed; inspect network, model access, and API credentials")
    probe_result = json.loads(process.stdout)
    (work / "live-probe.json").write_text(json.dumps(probe_result, indent=2))
    print(json.dumps(probe_result, indent=2))
else:
    print("Paid probe disabled. Kaggle compute does not provide an OpenAI API key.")
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.11"}},
                "cells": [cell("# BinanceDocTalk — private Kaggle smoke run\n\nIndependent educational project. No affiliation with Binance. CPU only; no paid calls by default.\n\nThis notebook embeds only allowlisted public project code and original draft/illustrative data. Official corpus and gold review remain separate."),
                          cell(bootstrap, True), cell(tests, True), cell(demo, True),
                          cell("## Optional live API probe\n\nKaggle authentication and OpenAI authentication are different. Set a Kaggle Secret only if you choose to run the paid provider probe. Do not place keys in notebook code. The CLI cannot create Kaggle Secrets."),
                          cell(probe, True),
                          cell("## Continue the official study\n\nFollow `docs/dataset_card.md` and `docs/evaluation.md` inside the extracted project. Verify collection/API processing rights, collect sources if permitted, manually review sources and gold, then tune on development and freeze. Do not run held-out test from unreviewed drafts. This smoke notebook intentionally does not do those steps automatically.\n\nOutputs: `/kaggle/working/core-tests.txt`, `benchmark-audit.json`, and `offline-demo.json`. A model quality report does not exist until real reviewed experiments are complete.")]}
    slug = "binancedoctalk-smoke-" + sha[:8]
    metadata = {"id": f"{owner}/{slug}", "title": "BinanceDocTalk Smoke " + sha[:8],
                "code_file": "binance_doctalk.ipynb", "language": "python", "kernel_type": "notebook",
                "is_private": True, "enable_gpu": False, "enable_internet": True,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    output.mkdir(parents=True, exist_ok=True)
    (output / "binance_doctalk.ipynb").write_text(json.dumps(notebook, indent=2) + "\n")
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"kernel": metadata["id"], "folder": str(output), "source_sha256": sha,
            "private": True, "gpu": False, "paid_api_calls_enabled": False, "remote_execution": "not_started"}


def prepare_gpu(owner: str, output: Path) -> dict:
    result = prepare(owner, output)
    path = output / "binance_doctalk.ipynb"
    notebook = json.loads(path.read_text())
    notebook["cells"][0] = cell("# BinanceDocTalk — private GPU integration\n\nPinned Qwen inference on original synthetic notes. NOT an official-source quality evaluation. No paid API calls.")
    install = """# Keep model caches out of notebook outputs; preserve Kaggle's CUDA-enabled PyTorch.
os.environ["HF_HOME"] = "/kaggle/temp/doctalk-huggingface"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DOCTALK_SOURCE_SHA256"] = EXPECTED_SHA256
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-local-model.txt"], check=True)
"""
    run = """# One process reuses the same model for baseline and RAG. Strict validation remains enabled.
subprocess.run([sys.executable, "scripts/gpu_smoke.py", str(work / "gpu-smoke.json")], check=True)
"""
    notebook["cells"] = notebook["cells"][:2] + [cell(install, True)] + notebook["cells"][2:4] + [cell(run, True),
        cell("Outputs: `gpu-smoke.json`, `core-tests.txt`, and `benchmark-audit.json`. GPU results use original synthetic notes and must not be cited as official-source benchmark scores.")]
    path.write_text(json.dumps(notebook, indent=2) + "\n")
    metadata_path = output / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(id=f"{owner}/binancedoctalk-gpu-{result['source_sha256'][:8]}",
                    title="BinanceDocTalk GPU " + result["source_sha256"][:8], enable_gpu=True,
                    machine_shape="NvidiaTeslaT4")
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    return {**result, "kernel": metadata["id"], "gpu": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "prepare", "push", "status", "output"])
    parser.add_argument("--owner")
    parser.add_argument("--gpu", action="store_true", help="Prepare private Qwen GPU integration notebook")
    parser.add_argument("--file-pattern", help="Limit downloaded outputs using a filename regex")
    parser.add_argument("--folder", type=Path, default=ROOT / "artifacts/local/kaggle/kernel")
    parser.add_argument("--check-auth", action="store_true")
    parser.add_argument("--download-to", type=Path, default=ROOT / "artifacts/local/kaggle/output")
    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(doctor(args.check_auth), indent=2))
        return
    env = environment()
    if args.command == "prepare":
        owner = args.owner or env.get("KAGGLE_USERNAME") or "YOUR_KAGGLE_USERNAME"
        print(json.dumps((prepare_gpu if args.gpu else prepare)(owner, args.folder), indent=2))
        return
    ready = doctor(check_auth=True)
    if not (ready["cli_available"] and ready["credentials_present"] and ready["dns_resolves"]
            and ready["authenticated"]):
        raise RuntimeError("Kaggle preflight failed: " + json.dumps(ready)
                           + ". Install requirements-kaggle.txt in a network-enabled environment. "
                           + "Existing valid OAuth credentials are reused; setup wizard is only needed if authentication is missing.")
    metadata_path = args.folder / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise ValueError("Prepare the kernel first")
    metadata = json.loads(metadata_path.read_text())
    if metadata["id"].startswith("YOUR_") or metadata.get("is_private") is not True:
        raise ValueError("Set a real owner and preserve private kernel visibility")
    command = cli_command()
    if args.command == "push":
        command += ["kernels", "push", "-p", str(args.folder)]
    elif args.command == "status":
        command += ["kernels", "status", metadata["id"]]
    else:
        command += ["kernels", "output", metadata["id"], "-p", str(args.download_to)]
        if args.file_pattern:
            command += ["--file-pattern", args.file_pattern]
    subprocess.run(command, env=cli_environment(env), check=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
