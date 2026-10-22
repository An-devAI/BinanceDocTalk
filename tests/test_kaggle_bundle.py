"""Kaggle preparation is tested locally; these tests never claim a remote execution."""

import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from package_source import source_archive
from kaggle_task import cli_command, cli_environment, doctor, prepare, prepare_gpu, prepare_embedding


class KaggleBundleTests(unittest.TestCase):
    def test_owner_username_does_not_block_oauth_with_incomplete_legacy_auth(self):
        self.assertNotIn("KAGGLE_USERNAME", cli_environment({"KAGGLE_USERNAME": "owner"}))
        legacy = {"KAGGLE_USERNAME": "owner", "KAGGLE_KEY": "synthetic"}
        self.assertEqual(cli_environment(legacy), legacy)

    def test_expired_oauth_with_refresh_token_can_attempt_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            (home / ".kaggle").mkdir()
            (home / ".kaggle/credentials.json").write_text(json.dumps({"refresh_token": "synthetic-test-value"}))
            with (patch("kaggle_task.Path.home", return_value=home),
                  patch("kaggle_task.environment", return_value={"KAGGLE_CONFIG_DIR": str(home / "config")}),
                  patch("kaggle_task.cli_command", return_value=["kaggle"]),
                  patch("kaggle_task.socket.getaddrinfo", return_value=[]),
                  patch("kaggle_task.subprocess.run", return_value=SimpleNamespace(returncode=0)) as run):
                result = doctor(check_auth=True)
                self.assertTrue(result["credentials_present"])
                self.assertTrue(result["authenticated"])
                run.assert_called_once()

    def test_project_kaggle_folder_is_not_mistaken_for_installed_cli(self):
        with (patch("kaggle_task.shutil.which", return_value=None),
              patch("kaggle_task.importlib.util.find_spec", return_value=SimpleNamespace(origin=None)),
              patch("kaggle_task.subprocess.run", return_value=SimpleNamespace(returncode=1))):
            self.assertIsNone(cli_command())

    def test_source_payload_excludes_credentials_private_data_and_venvs(self):
        payload = source_archive(False)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = archive.namelist()
        self.assertIn("src/binance_doctalk/answering.py", names)
        self.assertIn("data/eval/drafts/test.jsonl", names)
        for name in names:
            self.assertNotIn("data/local", name)
            self.assertNotIn("artifacts/local", name)
            self.assertNotIn(".venv", name)
            self.assertNotIn(".aws", name)
            self.assertNotIn(".env.kaggle", name)
            self.assertNotEqual(name, ".env")
            self.assertFalse(name.endswith(".ipynb"))

    def test_notebook_cells_compile_and_default_execution_is_private_unpaid_cpu(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = prepare("unit-test-owner", path)
            notebook = json.loads((path/"binance_doctalk.ipynb").read_text())
            metadata = json.loads((path/"kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertFalse(result["paid_api_calls_enabled"])
            self.assertEqual(result["remote_execution"], "not_started")
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), f"cell-{index}", "exec")
            bootstrap = "".join(notebook["cells"][1]["source"])
            # Parse string assignments, without executing the notebook or accessing the network.
            import ast
            values = {node.targets[0].id: node.value.value for node in ast.parse(bootstrap).body
                      if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                      and isinstance(node.targets[0], ast.Name)}
            raw = base64.b64decode(values["PAYLOAD"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), values["EXPECTED_SHA256"])

    def test_gpu_notebook_preserves_private_visibility_and_separates_smoke(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = prepare_gpu("unit-test-owner", path)
            notebook = json.loads((path/"binance_doctalk.ipynb").read_text())
            metadata = json.loads((path/"kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertTrue(metadata["enable_gpu"])
            self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
            self.assertEqual(metadata["id"], result["kernel"])
            for index, item in enumerate(notebook["cells"]):
                if item["cell_type"] == "code":
                    code = "".join(item["source"])
                    compile(code, f"gpu-{index}", "exec")
                    self.assertNotIn("scripts/doctalk.py evaluate", code)
            source = json.dumps(notebook)
            self.assertIn("/kaggle/temp", source)
            self.assertIn("scripts/gpu_smoke.py", source)
            self.assertNotIn("RUN_PAID_OPENAI_PROBE", source)

    def test_embedding_notebook_never_loads_generation_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            result = prepare_embedding("unit-test-owner", path)
            notebook = json.loads((path/"binance_doctalk.ipynb").read_text())
            metadata = json.loads((path/"kernel-metadata.json").read_text())
            self.assertTrue(metadata["is_private"])
            self.assertFalse(metadata["enable_gpu"])
            self.assertFalse(result["gpu"])
            self.assertNotIn("scripts/gpu_smoke.py", json.dumps(notebook))
            self.assertIn("scripts/embedding_smoke.py", json.dumps(notebook))
            for index,item in enumerate(notebook["cells"]):
                if item["cell_type"] == "code":
                    compile("".join(item["source"]), f"embedding-{index}", "exec")

    def test_email_or_token_cannot_be_used_as_owner(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                prepare("name@example.com", Path(folder))
