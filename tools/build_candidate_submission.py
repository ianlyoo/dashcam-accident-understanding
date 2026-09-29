"""Build a controlled candidate ZIP by appending one stage override."""

from __future__ import annotations

import argparse
import ast
import re
import zipfile
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("Baseline"))
    parser.add_argument(
        "--baseline-zip",
        type=Path,
        help="Use an already validated submission ZIP as the immutable base",
    )
    parser.add_argument("--override", type=Path, action="append", required=True)
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3"),
        action="append",
        required=True,
    )
    parser.add_argument(
        "--isolate-stage2-collision",
        action="store_true",
        help=(
            "Execute a single Stage 2 override in a private namespace and "
            "replace only the baseline collision_frame column"
        ),
    )
    parser.add_argument(
        "--isolate-stage2-fields",
        nargs="+",
        choices=(
            "collision_frame",
            "entry_frame",
            "evasion_space",
            "entry_side",
        ),
        help=(
            "Execute a single Stage 2 override in a private namespace and "
            "replace only these baseline output fields"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--extra-file",
        nargs=2,
        action="append",
        metavar=("SOURCE", "ARCHIVE_PATH"),
        help="Add a local model asset at a model/... path inside the ZIP",
    )
    args = parser.parse_args()

    baseline = args.baseline_dir.resolve()
    archived_members = None
    if args.baseline_zip is None:
        base_source = (baseline / "inference.py").read_text(encoding="utf-8")
    else:
        baseline_zip = args.baseline_zip.resolve()
        if baseline_zip == args.output.resolve():
            raise RuntimeError("--baseline-zip and --output must be different files")
        with zipfile.ZipFile(baseline_zip) as archive:
            base_source = archive.read("inference.py").decode("utf-8")
            archived_members = [
                (info, archive.read(info.filename))
                for info in archive.infolist()
                if not info.is_dir() and info.filename != "inference.py"
            ]
    if len(args.override) != len(args.stage):
        raise RuntimeError("provide one --override for each --stage")
    override_parts = []
    for path in args.override:
        override_source = path.read_text(encoding="utf-8")
        override_parts.append(
            re.sub(r"(?m)^from __future__ import [^\n]+\n?", "", override_source)
        )
    extra_files = []
    for source_text, archive_text in args.extra_file or []:
        source_path = Path(source_text)
        archive_path = PurePosixPath(archive_text.replace("\\", "/"))
        if not source_path.is_file():
            raise RuntimeError(f"extra file not found: {source_path}")
        if (
            archive_path.is_absolute()
            or ".." in archive_path.parts
            or not archive_path.parts
            or archive_path.parts[0] != "model"
        ):
            raise RuntimeError(
                f"extra archive path must be safe and under model/: {archive_text}"
            )
        extra_files.append((source_path, archive_path.as_posix()))
    source = base_source.rstrip()
    if args.isolate_stage2_collision and args.isolate_stage2_fields:
        raise RuntimeError(
            "use either --isolate-stage2-collision or --isolate-stage2-fields"
        )
    isolated_fields = (
        ["collision_frame"]
        if args.isolate_stage2_collision
        else args.isolate_stage2_fields
    )
    if isolated_fields:
        if args.stage != ["stage2"] or len(override_parts) != 1:
            raise RuntimeError(
                "Stage 2 field isolation requires exactly one Stage 2 override"
            )
        candidate_source = override_parts[0]
        field_tuple = tuple(dict.fromkeys(isolated_fields))
        source += f"""

# ISOLATED STAGE 2 FIELD OVERRIDE
_S2_CHAMPION_PREDICT_STAGE2 = predict_stage2
_S2_CANDIDATE_NAMESPACE = {{"__name__": "_s2_field_candidate"}}
exec(
    compile({candidate_source!r}, "s2_field_candidate.py", "exec"),
    _S2_CANDIDATE_NAMESPACE,
)
_S2_CANDIDATE_PREDICT_STAGE2 = _S2_CANDIDATE_NAMESPACE["predict_stage2"]
_S2_REPLACE_FIELDS = {field_tuple!r}


def predict_stage2(data_dir, model_dir):
    baseline = _S2_CHAMPION_PREDICT_STAGE2(data_dir, model_dir).copy()
    candidate = _S2_CANDIDATE_PREDICT_STAGE2(data_dir, model_dir)
    for field in _S2_REPLACE_FIELDS:
        candidate_by_id = dict(zip(candidate["ID"].astype(str), candidate[field]))
        baseline[field] = [
            candidate_by_id.get(str(sample_id), old_value)
            for sample_id, old_value in zip(baseline["ID"], baseline[field])
        ]
    return baseline
"""
    else:
        for override_source in override_parts:
            source += (
                "\n\n# CONTROLLED CANDIDATE OVERRIDE\n"
                + override_source.rstrip()
                + "\n"
            )
    tree = ast.parse(source, filename="candidate/inference.py")
    compile(tree, "candidate/inference.py", "exec")
    for stage in args.stage:
        target = f"predict_{stage}"
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == target
        ]
        if len(definitions) < 2:
            raise RuntimeError(f"override must redefine {target}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.writestr("inference.py", source)
        if archived_members is not None:
            existing_names = {info.filename for info, _payload in archived_members}
            for info, payload in archived_members:
                archive.writestr(info, payload)
            for source_path, archive_path in extra_files:
                if archive_path in existing_names:
                    raise RuntimeError(f"extra archive path already exists: {archive_path}")
                archive.write(source_path, archive_path)
        else:
            archive.write(baseline / "requirements.txt", "requirements.txt")
            for model_path in sorted((baseline / "model").rglob("*")):
                if model_path.is_file():
                    archive.write(
                        model_path, model_path.relative_to(baseline).as_posix()
                    )
            for source_path, archive_path in extra_files:
                archive.write(source_path, archive_path)
    print(f"built {args.output} ({args.output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
