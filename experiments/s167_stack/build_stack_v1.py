"""Deterministic base-relative arm merge. Refuse unresolved overlapping edits."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
BASE_SHA = 'e7fb70a66ad251d47c180861e2e001da7b270259c2e12ca394c6de3f1c510d70'
KNOWN = {'S163_s1.zip': '0fb0c736473eeb56eacfb50e18d3fcf1ca49c54f1e0c68afe97d25526853a5d8',
         'S164_fast.zip': 'ab02bf4c944a094ac4c9eba4b401c27e43fb9d519422c6876d8d17feb342fc02'}

def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(chunk)
    return h.hexdigest()

def sha(path):
    with Path(path).open('rb') as stream:
        return digest(stream)

def members(archive):
    result = {}
    for info in archive.infolist():
        name = info.filename
        assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
        assert '\\' not in name and ':' not in name
        if info.is_dir():
            continue
        assert name not in result, name
        with archive.open(info) as stream:
            result[name] = digest(stream)
    return result

def merge_text(base, versions, name):
    # All edits are anchored to the same base, never to a previously merged arm.
    lines = base.decode('utf-8').splitlines(keepends=True)
    edits = []
    for owner, content in versions:
        new = content.decode('utf-8').splitlines(keepends=True)
        for tag, start, end, a, b in difflib.SequenceMatcher(None, lines, new, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            replacement = new[a:b]
            duplicate = False
            for old_start, old_end, old_replacement, old_owner in edits:
                if (start, end, replacement) == (old_start, old_end, old_replacement):
                    duplicate = True
                    break
                # Separate EOF append dispatches are composed in declared order.
                if start == end == old_start == old_end == len(lines):
                    continue
                overlap = max(start, old_start) < min(end, old_end)
                overlap |= start == end and old_start <= start < old_end
                overlap |= old_start == old_end and start <= old_start < end
                overlap |= start == end == old_start == old_end
                if overlap:
                    raise ValueError(f'Manual overlap resolution required: {name}: {owner} / {old_owner}')
            if not duplicate:
                edits.append((start, end, replacement, owner))
    result, cursor = [], 0
    for start, end, replacement, owner in sorted(edits, key=lambda x: (x[0], x[1])):
        result.extend(lines[cursor:start])
        result.extend(replacement)
        cursor = end
    result.extend(lines[cursor:])
    return ''.join(result).encode('utf-8')

def stage(args):
    base = Path(args.base)
    assert sha(base) == BASE_SHA, 'Wrong S156 carrier'
    arms = [Path(p) for p in args.arms]
    assert len(set(p.resolve() for p in arms)) == len(arms)
    # Runtime dispatcher must wrap all other arms' final functions.
    arms.sort(key=lambda p: p.name == 'S164_fast.zip')
    root = Path(args.out)
    assert not root.exists(), 'Preserve existing tree; use a new --out for phase 2'
    report = dict(base=str(base), base_sha256=BASE_SHA, arms=[], overlaps={}, members={})
    archives = []
    try:
        bz = zipfile.ZipFile(base)
        archives.append(bz)
        before = members(bz)
        deltas = {}
        for arm in arms:
            identity = sha(arm)
            if arm.name in KNOWN:
                assert identity == KNOWN[arm.name], arm
            az = zipfile.ZipFile(arm)
            archives.append(az)
            current = members(az)
            assert not set(before) - set(current), 'Arm deletes base files; manual review required'
            changed = sorted(k for k in before if before[k] != current[k])
            added = sorted(set(current) - set(before))
            report['arms'].append(dict(path=str(arm), sha256=identity, changed=changed, added=added,
                                       member_sha256=current))
            for name in changed + added:
                deltas.setdefault(name, []).append((arm.name, az.read(name)))
            print(arm.name, 'changed', changed, 'added', added, flush=True)
        root.mkdir(parents=True)
        for name in sorted(set(before) | set(deltas)):
            target = root / name
            assert target.resolve().is_relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            versions = deltas.get(name, [])
            if not versions:
                with bz.open(name) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
            else:
                contents = {v for _, v in versions}
                if len(contents) == 1:
                    merged = versions[0][1]
                else:
                    assert name in before and name.endswith('.py'), 'Unsupported overlapping file: ' + name
                    merged = merge_text(bz.read(name), versions, name)
                    report['overlaps'][name] = dict(arms=[o for o, _ in versions],
                        resolution='Base-relative non-overlapping edits; EOF appends in listed order, S164 last')
                if name.endswith('.py'):
                    compile(merged, name, 'exec')
                target.write_bytes(merged)
            report['members'][name] = sha(target)
        report['changed_from_base'] = sorted(k for k in before if before[k] != report['members'][k])
        report['added_to_base'] = sorted(set(report['members']) - set(before))
        report['unchanged_base_members'] = len(before) - len(report['changed_from_base'])
        report['build_script_sha256'] = sha(__file__)
        (root.parent / 'build.json').write_text(json.dumps(report, indent=2) + '\n')
        print('STAGED', len(report['members']), 'members', flush=True)
    finally:
        for archive in archives:
            archive.close()

def package(args):
    root, dest = Path(args.out), Path(args.zip)
    report = json.loads((root.parent / 'build.json').read_text())
    actual = {p.relative_to(root).as_posix(): sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual == report['members'], 'Staged files changed'
    assert dest.name.isascii() and len(dest.name) <= 30
    assert not dest.exists(), 'Preserve existing release'
    with zipfile.ZipFile(dest, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name in sorted(actual):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info._compresslevel = 1
            info.external_attr = 0o100644 << 16
            with archive.open(info, 'w', force_zip64=True) as dst, (root / name).open('rb') as src:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    with zipfile.ZipFile(dest) as archive:
        assert members(archive) == actual  # streams every member, checking CRC and SHA256
    receipt = dict(path=str(dest), sha256=sha(dest), bytes=dest.stat().st_size,
                   members=len(actual), crc_and_member_hashes_passed=True,
                   build_sha256=sha(root.parent / 'build.json'), release_ready=False)
    (root.parent / 'archive.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='$DATA_DIR/releases/S156_casc.zip')
    parser.add_argument('--arms', nargs='+', required=True)
    parser.add_argument('--out', default='$DATA_DIR/s167_stack/phase1/candidate')
    parser.add_argument('--zip', default='$DATA_DIR/releases/S167_stk.zip')
    parser.add_argument('--action', choices=['stage', 'package', 'all'], default='all')
    args = parser.parse_args()
    if args.action in ('stage', 'all'):
        stage(args)
    if args.action in ('package', 'all'):
        package(args)
