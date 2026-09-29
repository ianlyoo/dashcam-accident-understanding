"""Static, offline checks for a DACON 236753 submission ZIP."""

from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path, PurePosixPath

MAX_ZIP_BYTES = 10 * 1024**3
MAX_UNPACKED_BYTES = 32 * 1024**3
MAX_FILENAME_CHARS = 30
ALLOWED_ROOTS = {"model", "inference.py", "requirements.txt"}
REQUIRED_FILES = {"inference.py", "requirements.txt"}
REQUIRED_FUNCTIONS = {"predict_stage1", "predict_stage2", "predict_stage3"}
REQUIRED_MODEL_DIRS = {"model/stage1", "model/stage2", "model/stage3"}


def _positional_signature_error(node: ast.AST) -> str | None:
    """Explain why node cannot be called as fn(data_dir, model_dir), else None.

    The official harness calls each entry point synchronously with exactly two
    positional arguments. Defaults, *args and **kwargs are acceptable as long as
    two positional arguments are accepted and nothing else is required.
    """
    if isinstance(node, ast.AsyncFunctionDef):
        return "is async, so calling it returns a coroutine instead of predictions"

    args = node.args
    positional = [*args.posonlyargs, *args.args]
    required_positional = len(positional) - len(args.defaults)
    if required_positional > 2:
        names = [arg.arg for arg in positional[:required_positional]]
        return f"requires {required_positional} positional arguments {names}"
    if len(positional) < 2 and args.vararg is None:
        return f"accepts at most {len(positional)} positional argument(s)"

    required_kwonly = [
        arg.arg
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
        if default is None
    ]
    if required_kwonly:
        return f"requires keyword-only argument(s) {required_kwonly}"
    return None


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    if len(path.name) > MAX_FILENAME_CHARS:
        errors.append(
            f"ZIP filename exceeds {MAX_FILENAME_CHARS} characters including extension "
            f"({len(path.name)}): {path.name}"
        )
    if not path.is_file():
        return [*errors, f"ZIP not found: {path}"]
    if path.stat().st_size > MAX_ZIP_BYTES:
        errors.append("compressed size exceeds 10 GiB")
    if not zipfile.is_zipfile(path):
        return [*errors, "file is not a valid ZIP archive"]

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename.replace("\\", "/") for info in infos]
        if len(names) != len(set(names)):
            errors.append("ZIP contains duplicate member names")
        if sum(info.file_size for info in infos) > MAX_UNPACKED_BYTES:
            errors.append("uncompressed size exceeds 32 GiB")

        normalized_files: set[str] = set()
        for name, info in zip(names, infos):
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                errors.append(f"unsafe archive path: {name}")
                continue
            if not member.parts:
                continue
            if member.parts[0] not in ALLOWED_ROOTS:
                errors.append(f"unexpected top-level member: {name}")
            if not info.is_dir():
                normalized_files.add(member.as_posix())

        missing = sorted(REQUIRED_FILES - normalized_files)
        if missing:
            errors.append(f"missing required files: {missing}")
        for model_dir in sorted(REQUIRED_MODEL_DIRS):
            prefix = model_dir + "/"
            if not any(name.startswith(prefix) for name in normalized_files):
                errors.append(f"no model file found under {model_dir}/")

        if "inference.py" in normalized_files:
            try:
                source = archive.read("inference.py").decode("utf-8")
                tree = ast.parse(source, filename="submit.zip/inference.py")
            except (UnicodeDecodeError, SyntaxError) as exc:
                errors.append(f"inference.py cannot be parsed as UTF-8 Python: {exc}")
            else:
                functions: dict[str, ast.AST] = {
                    node.name: node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                absent = sorted(REQUIRED_FUNCTIONS - set(functions))
                if absent:
                    errors.append(f"missing required top-level functions: {absent}")
                for name in sorted(REQUIRED_FUNCTIONS & set(functions)):
                    reason = _positional_signature_error(functions[name])
                    if reason is not None:
                        errors.append(
                            f"{name} is not callable as {name}(data_dir, model_dir): {reason}"
                        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    args = parser.parse_args()
    errors = validate(args.zip_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"OK: {args.zip_path}")


if __name__ == "__main__":
    main()
