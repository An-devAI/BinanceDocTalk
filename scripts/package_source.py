"""Package only explicitly public project assets; never walk credentials/local data."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def public_files(include_media: bool = True) -> list[Path]:
    names = ["README.md", "LICENSE", "NOTICE.md", "pyproject.toml", "uv.lock", ".gitignore",
             ".env.example", "app.py", "web_app.py", "requirements-ui.txt",
             "requirements-tokenizer.txt", "requirements-kaggle.txt", "requirements-local-model.txt", "data/LICENSE.md",
             "data/manifest.v1.jsonl", "artifacts/results/README.md", ".github/workflows/ci.yml"]
    paths = [ROOT / name for name in names if (ROOT / name).is_file()]
    for pattern in ["src/binance_doctalk/*.py", "src/binance_doctalk/ui.html", "tests/*.py",
                    "scripts/*.py", "scripts/*.sh", "configs/*.toml", "docs/*.md", "docs/kaggle_run.json",
                    "data/demo/*.jsonl", "data/eval/drafts/*.jsonl", "kaggle/README.md"]:
        paths.extend(ROOT.glob(pattern))
    if include_media:
        for pattern in ["demo/*.md", "demo/*.json", "demo/*.html", "demo/*.png", "demo/*.webm",
                        "kaggle/*.ipynb", "kaggle/*.template.json"]:
            paths.extend(ROOT.glob(pattern))
    return sorted(set(path for path in paths if path.is_file()))


def source_archive(include_media: bool = True) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in public_files(include_media):
            # Stable archive timestamps make its digest independent of filesystem mtimes.
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return output.getvalue()


if __name__ == "__main__":
    target = ROOT / "dist/binance-doctalk-source.zip"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(source_archive())
    print(f"Created {target.name}; only allowlisted public files were included.")
