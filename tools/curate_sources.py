"""Copy the publishable source subset from a local competition archive.

Set SOURCE_REPO and FINAL_RELEASE_ZIP before running. No source is modified.
The resulting tree still needs a human privacy and licence review.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ["SOURCE_REPO"])
RELEASE = Path(os.environ["FINAL_RELEASE_ZIP"])


def clean(value: str) -> str:
    value = re.sub(r"/mnt/c/Users/[^/]+/Documents/code/videohackathon", ".", value)
    value = re.sub(r"(?i)[A-Z]:[/\\]Users[/\\][^/\\\s'\"]+[/\\]Documents[/\\]code[/\\]videohackathon", ".", value)
    value = re.sub(r"(?i)[A-Z]:[/\\]videohackathon_data", "$DATA_DIR", value)
    value = re.sub(r"(?i)[A-Z]:[/\\]Users[/\\][^/\\\s'\"]+", "$USER_HOME", value)
    value = re.sub(r"(?i)[A-Z]:/Program Files/Git/cmd/git.exe", "git", value)
    value = re.sub(r"(?i)[A-Z]:/Windows/System32/(nvidia-smi|wsl).exe", r"\1", value)
    value = re.sub(r"(?i)[A-Z]:/", "$LOCAL_ROOT/", value)
    value = re.sub(r"/mnt/c/Users/[^/]+", "$USER_HOME", value)
    return value


def write_text(dest: Path, raw: bytes) -> None:
    text = raw.decode("utf-8", errors="replace")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(clean(text), encoding="utf-8")


with zipfile.ZipFile(RELEASE) as archive:
    for member in archive.infolist():
        name = member.filename
        if member.is_dir() or ".." in Path(name).parts:
            continue
        path = Path(name)
        if path.suffix.lower() not in {".py", ".txt", ".md", ".json"}:
            continue
        if member.file_size > 100_000 and path.suffix.lower() == ".json":
            continue  # learned parameters encoded as JSON
        if member.file_size > 1_000_000:
            continue
        write_text(ROOT / "src" / "final_pipeline" / path, archive.read(member))

for candidate in (
    "s160_coll_loc", "s161_entry_cross", "s167_stack", "s172_vis_coll",
    "s174_cov", "s176_bucket", "s178_side", "s180_resid", "s181_fallback",
):
    folder = SOURCE / "candidates" / candidate
    for path in folder.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if any(part in {"source", "harness_before_atomic", "harness_before_fault"} for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".patch", ".diff", ".md"}:
            continue
        if path.stat().st_size > 1_000_000:
            continue
        dest = ROOT / "experiments" / candidate / path.relative_to(folder)
        write_text(dest, path.read_bytes())

for name in ("evaluation.py", "__init__.py"):
    path = SOURCE / "src" / "videohackathon" / name
    write_text(ROOT / "src" / "videohackathon" / name, path.read_bytes())

for name in ("validate_submission.py", "build_candidate_submission.py", "run_baseline_smoke.py"):
    path = SOURCE / "tools" / name
    write_text(ROOT / "tools" / name, path.read_bytes())

for path in (SOURCE / "tests").rglob("test_*.py"):
    if path.stat().st_size <= 100_000:
        write_text(ROOT / "tests" / path.relative_to(SOURCE / "tests"), path.read_bytes())
