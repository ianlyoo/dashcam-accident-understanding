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
                    normalize = name in getattr(args, 'normalize_merge_path', [])
                    source = bz.read(name)
                    if normalize:
                        source = source.replace(b'\r\n', b'\n')
                        versions = [(owner, content.replace(b'\r\n', b'\n')) for owner, content in versions]
                    merged = merge_text(source, versions, name)
                    report['overlaps'][name] = dict(arms=[o for o, _ in versions],
                        normalized_crlf=normalize,
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


def overlay(args):
    """Overlay an arm based on S171 onto the exact S172 carrier."""
    assert args.carrier and len(args.arms) == 1
    base, carrier, arm = map(Path, (args.base, args.carrier, args.arms[0]))
    pinned = {
        base.name: 'f2f3b55617d80fdee1afabe930d426bc92c99ceb7f48f2953e6504efda280315',
        carrier.name: '2969891d7472a8fb8bcc6195fee97bf4f55185163149bf71869b5c46fdced549',
        arm.name: 'd2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc',
    }
    assert (base.name, carrier.name, arm.name) == ('S171_ent.zip','S172_vis.zip','S174_cov.zip')
    for path in (base,carrier,arm):
        assert sha(path) == pinned[path.name], path
    root = Path(args.out)
    assert not root.exists(), 'Preserve existing staged tree'
    with zipfile.ZipFile(base) as bz, zipfile.ZipFile(carrier) as cz, zipfile.ZipFile(arm) as az:
        before, carrier_members, arm_members = members(bz), members(cz), members(az)
        assert set(before) <= set(carrier_members) and set(before) <= set(arm_members)
        carrier_changed = {k for k in before if carrier_members[k] != before[k]}
        arm_changed = {k for k in before if arm_members[k] != before[k]}
        carrier_added = set(carrier_members) - set(before)
        arm_added = set(arm_members) - set(before)
        assert carrier_changed == {'inference.py','model/stage2/s144/predict.py'}
        assert arm_changed == {'model/stage2/s161/predict.py'}
        assert arm_added == {'model/licenses/S174-NOTICE.txt'}
        assert not (carrier_changed & arm_changed or carrier_added & arm_added)
        assert not set(before) - set(carrier_members) and not set(before) - set(arm_members)
        root.mkdir(parents=True)
        expected = {}
        for name in sorted(set(carrier_members) | arm_added):
            target=root/name
            assert target.resolve().is_relative_to(root.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            source=az if name in arm_changed | arm_added else cz
            with source.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src,dst,2**20)
            expected[name]=sha(target)
        # The final dispatch follows S172's visual collision and applies S174
        # once more on that final collision. S172 rows are the error fallback.
        wrapper='''\n\n# S175: S172 collision, then S174 entry on that final anchor, then clamp.\n_S175_BASE_STAGE2 = predict_stage2\n\ndef predict_stage2(data_dir, model_dir):\n    import importlib.util\n    from pathlib import Path\n    base = _S175_BASE_STAGE2(data_dir, model_dir)\n    try:\n        source = Path(__file__).resolve().parent / 'model/stage2/s175_entry.py'\n        spec = importlib.util.spec_from_file_location('_s175_entry', source)\n        module = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(module)\n        return module.apply(globals(), base, data_dir, model_dir)\n    except Exception as exc:\n        globals()['_S175_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}\n        return base\n'''
        inference=root/'inference.py'
        assert inference.read_bytes().endswith(b'        return base\n'), 'Unexpected S172 dispatcher suffix'
        inference.write_bytes(inference.read_bytes()+wrapper.encode())
        runtime = HERE/'phase4/s175_entry.py'
        target=root/'model/stage2/s175_entry.py'
        shutil.copyfile(runtime,target)
        expected['inference.py']=sha(inference)
        expected['model/stage2/s175_entry.py']=sha(target)
        for path in root.rglob('*.py'):
            compile(path.read_bytes(),str(path),'exec')
        report=dict(base=str(base),base_sha256=sha(base),carrier=str(carrier),
                    carrier_sha256=sha(carrier),arm=str(arm),arm_sha256=sha(arm),
                    carrier_changed=sorted(carrier_changed),carrier_added=sorted(carrier_added),
                    arm_changed=sorted(arm_changed),arm_added=sorted(arm_added),
                    changed_from_carrier=['inference.py',*sorted(arm_changed)],
                    added_to_carrier=sorted(arm_added | {'model/stage2/s175_entry.py'}),
                    members=expected,build_script_sha256=sha(__file__),
                    postprocessor_sha256=sha(runtime),
                    order='S172 final collision -> S174 S161/coverage on final collision -> min(entry, collision); error restores exact S172 entry')
        (root.parent/'build.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='members'},indent=2),flush=True)


def overlay_bucket(args):
    """Keep exact S175 behavior, then add S176's two abstention buckets."""
    assert args.carrier and len(args.arms) == 1
    base, carrier, arm = map(Path, (args.base, args.carrier, args.arms[0]))
    pinned = {
        'S174_cov.zip': 'd2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc',
        'S175_stk.zip': '3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be',
        'S176_bucket.zip': 'fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68',
    }
    assert (base.name, carrier.name, arm.name) == tuple(pinned)
    for path in (base, carrier, arm):
        assert sha(path) == pinned[path.name], path
    root = Path(args.out)
    assert not root.exists(), 'Preserve existing staged tree'
    with zipfile.ZipFile(base) as bz, zipfile.ZipFile(carrier) as cz, zipfile.ZipFile(arm) as az:
        common, carrier_members, arm_members = members(bz), members(cz), members(az)
        assert set(common) <= set(carrier_members) and set(common) <= set(arm_members)
        changed = {name for name in common if arm_members[name] != common[name]}
        added = set(arm_members) - set(common)
        deleted = set(common) - set(arm_members)
        assert changed == {'model/stage2/s161/predict.py'}
        assert added == {'model/licenses/S176-NOTICE.txt'} and not deleted
        assert carrier_members['model/stage2/s161/predict.py'] == common['model/stage2/s161/predict.py']
        source = az.read('model/stage2/s161/predict.py')
        assert source.startswith(bz.read('model/stage2/s161/predict.py'))
        root.mkdir(parents=True)
        expected = {}
        for name in sorted(carrier_members):
            target = root / name
            assert target.resolve().is_relative_to(root.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with cz.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, 2**20)
            expected[name] = sha(target)
        new_sources = {
            'model/stage2/s176_predict.py': source,
            'model/licenses/S176-NOTICE.txt': az.read('model/licenses/S176-NOTICE.txt'),
            'model/stage2/s177_entry.py': (HERE/'phase5/s177_entry.py').read_bytes(),
        }
        for name, content in new_sources.items():
            assert name not in expected
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            expected[name] = sha(target)
        wrapper = '''\n\n# S177: S175 rows, then only S176 abstention buckets on final S172 collision.\n_S177_BASE_STAGE2 = predict_stage2\n\ndef predict_stage2(data_dir, model_dir):\n    import importlib.util\n    from pathlib import Path\n    base = _S177_BASE_STAGE2(data_dir, model_dir)\n    try:\n        source = Path(__file__).resolve().parent / 'model/stage2/s177_entry.py'\n        spec = importlib.util.spec_from_file_location('_s177_entry', source)\n        module = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(module)\n        return module.apply(globals(), base, data_dir, model_dir)\n    except Exception as exc:\n        globals()['_S177_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}\n        return base\n'''
        inference = root/'inference.py'
        assert inference.read_bytes().endswith(b'        return base\n'), 'Unexpected S175 dispatcher suffix'
        inference.write_bytes(inference.read_bytes() + wrapper.encode())
        expected['inference.py'] = sha(inference)
        for path in root.rglob('*.py'):
            compile(path.read_bytes(), str(path), 'exec')
        report = dict(base=str(base), base_sha256=sha(base), carrier=str(carrier),
                      carrier_sha256=sha(carrier), arm=str(arm), arm_sha256=sha(arm),
                      arm_changed=sorted(changed), arm_added=sorted(added),
                      changed_from_carrier=['inference.py'],
                      added_to_carrier=sorted(new_sources), members=expected,
                      build_script_sha256=sha(__file__),
                      postprocessor_sha256=sha(HERE/'phase5/s177_entry.py'),
                      order='S175 exact base -> S176 bucket-only on S172 collision -> clamp; error retains exact S175 entry')
        (root.parent/'build.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='members'}, indent=2), flush=True)


def overlay_chain(args):
    """Replay ordered S176-lineage arms after exact S177's S172-anchored rows."""
    carrier = Path(args.carrier)
    assert carrier.name == 'S177_bkt.zip'
    assert sha(carrier) == 'b30e56829cfbbad98bd3121b1c7ced6760fb8bb1da97f73aba006d45cb26adea'
    arms = [Path(p) for p in args.arms]
    full = len(arms) == 3
    assert [p.name for p in arms] == (['S178_side.zip', 'S180_resid.zip', 'S181_fallback.zip']
                                      if full else ['S178_side.zip'])
    pins = {
        'S178_side.zip': 'bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047',
        'S180_resid.zip': 'd5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b',
        'S181_fallback.zip': '547323b4f5a5c34ee273c67ac2cba91694844ea179bda248aa870c4b89548b77',
    }
    for arm in arms:
        assert sha(arm) == pins[arm.name], arm
    root = Path(args.out)
    assert not root.exists(), 'Preserve existing staged tree'
    model = 'model/stage2/s161/predict.py'
    adapter = 'model/stage2/s118/adapter.py'
    notices = ['model/licenses/S178-NOTICE.txt', 'model/licenses/S180-NOTICE.txt',
               'model/licenses/S181-NOTICE.txt']
    with zipfile.ZipFile(carrier) as cz, zipfile.ZipFile(
            '$DATA_DIR/releases/S176_bucket.zip') as bz:
        assert sha('$DATA_DIR/releases/S176_bucket.zip') == \
            'fc5c3da13c1c0f00b61a73377d55c4d548d12c26031d0837aa042c3f61abda68'
        carrier_members = members(cz)
        assert hashlib.sha256(bz.read(model)).hexdigest() == carrier_members['model/stage2/s176_predict.py']
        previous = bz.read(model)
        previous_members = members(bz)
        arm_sources = []
        with zipfile.ZipFile(arms[0]) as az:
            arm_members = members(az)
            changed = {k for k in previous_members if arm_members[k] != previous_members[k]}
            added = set(arm_members) - set(previous_members)
            assert changed == {model, adapter} and added == {notices[0]}
            assert az.read(model).startswith(previous)
            adapter_source = az.read(adapter)
            arm_sources.append(az.read(model))
            notice_bytes = {notices[0]: az.read(notices[0])}
            previous_members = arm_members
            previous = arm_sources[-1]
        for index, arm in enumerate(arms[1:], start=1):
            with zipfile.ZipFile(arm) as az:
                arm_members = members(az)
                changed = {k for k in previous_members if arm_members[k] != previous_members[k]}
                added = set(arm_members) - set(previous_members)
                assert changed == {model} and added == {notices[index]}
                assert az.read(model).startswith(previous)
                assert az.read(adapter) == adapter_source
                previous = az.read(model)
                arm_sources.append(previous)
                notice_bytes[notices[index]] = az.read(notices[index])
                previous_members = arm_members
        root.mkdir(parents=True)
        expected = {}
        for name in sorted(carrier_members):
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with cz.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, 2**20)
            expected[name] = sha(target)
        new_sources = {
            'model/stage2/s181_predict.py' if full else 'model/stage2/s178_predict.py': arm_sources[-1],
            'model/stage2/s178_adapter.py': adapter_source,
            'model/stage2/s182_chain.py' if full else 'model/stage2/s179_chain.py':
                (HERE/'phase6/chain_entry.py').read_bytes(),
            **notice_bytes,
        }
        for name, content in new_sources.items():
            assert name not in expected
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            expected[name] = sha(target)
        label = 'S182' if full else 'S179'
        runtime = 's182_chain.py' if full else 's179_chain.py'
        wrapper = f'''\n\n# {label}: ordered chained rules on exact S177 output and final S172 collision.\n_{label}_BASE_STAGE2 = predict_stage2\n\ndef predict_stage2(data_dir, model_dir):\n    import importlib.util\n    from pathlib import Path\n    base = _{label}_BASE_STAGE2(data_dir, model_dir)\n    try:\n        source = Path(__file__).resolve().parent / 'model/stage2/{runtime}'\n        spec = importlib.util.spec_from_file_location('_{label.lower()}_chain', source)\n        module = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(module)\n        return module.apply(globals(), base, data_dir, model_dir, full={full})\n    except Exception as exc:\n        globals()['_{label}_DIAGNOSTICS'] = {{'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {{}}}}\n        return base\n'''
        inference = root/'inference.py'
        assert inference.read_bytes().endswith(b'        return base\n')
        inference.write_bytes(inference.read_bytes() + wrapper.encode())
        expected['inference.py'] = sha(inference)
        for path in root.rglob('*.py'):
            compile(path.read_bytes(), str(path), 'exec')
        report = dict(carrier=str(carrier), carrier_sha256=sha(carrier),
                      arms=[dict(path=str(p), sha256=pins[p.name]) for p in arms],
                      source_chain_sha256=[hashlib.sha256(s).hexdigest() for s in arm_sources],
                      stage2_anchor='S172 final collision',
                      changed_from_carrier=['inference.py'],
                      added_to_carrier=sorted(new_sources), members=expected,
                      build_script_sha256=sha(__file__),
                      postprocessor_sha256=sha(HERE/'phase6/chain_entry.py'))
        (root.parent/'build.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='members'}, indent=2), flush=True)


def overlay_nobucket(args):
    """Port S178/S180/S181 non-bucket rules onto exact S175/S174."""
    carrier = Path(args.carrier)
    assert carrier.name == 'S175_stk.zip'
    assert sha(carrier) == '3827273f061a7222e22d693207d1f5897bcefe114b20802adf48c971ff1e98be'
    arms = [Path(p) for p in args.arms]
    assert [p.name for p in arms] == ['S178_side.zip', 'S180_resid.zip', 'S181_fallback.zip']
    pins = {
        'S178_side.zip': 'bab785c758b82f7524bdaa1153220c3aa61a7d8b120ae5cac14c049ec83ec047',
        'S180_resid.zip': 'd5033adfce1d40fb5fc98289196fb791b274ad4529d111c3d2fbacaa9623a84b',
        'S181_fallback.zip': '547323b4f5a5c34ee273c67ac2cba91694844ea179bda248aa870c4b89548b77',
    }
    for arm in arms:
        assert sha(arm) == pins[arm.name], arm
    patches = {
        'S178_on_S176.patch': '0b380dc599a2a68d9a3a7118e43e0f3296b7c106f8ce295fff2be133cdf31c50',
        'S180_on_S178.patch': '3931fe3e86140b08387d2f04b480fbd5aa039bda624133b587abc64caa736d8d',
        'S181_on_S180.patch': 'd5b728da9e378f90beb49808595ee04fd465f03a088695cc6be570f1be0db363',
    }
    patch_paths = [HERE.parent/'s178_side/S178_on_S176.patch',
                   HERE.parent/'s180_resid/S180_on_S178.patch',
                   HERE.parent/'s181_fallback/S181_on_S180.patch']
    for path in patch_paths:
        assert sha(path) == patches[path.name], path
    root = Path(args.out)
    assert not root.exists(), 'Preserve existing staged tree'
    model = 'model/stage2/s161/predict.py'
    with zipfile.ZipFile(carrier) as cz, zipfile.ZipFile(
            '$DATA_DIR/releases/S174_cov.zip') as az:
        assert sha('$DATA_DIR/releases/S174_cov.zip') == \
            'd2af1b1c63625035aa99199b0cebbb4d342eba7425190f75b9f2a6c49e182bdc'
        carrier_members = members(cz)
        assert not any('s176' in name.lower() or 'S176' in name for name in carrier_members)
        source = cz.read(model)
        assert source == az.read(model), 'S175 must contain exact S174 tracker'
        assert b'_s176_analyze' not in source
        extension = (HERE/'phase8/s185_extension.py').read_bytes()
        runtime = (HERE/'phase8/s185_postprocess.py').read_bytes()
        combined = source + extension
        compile(combined, 'model/stage2/s185_predict.py', 'exec')
        root.mkdir(parents=True)
        expected = {}
        for name in sorted(carrier_members):
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            with cz.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst, 2**20)
            expected[name] = sha(target)
        new_sources = {
            'model/stage2/s185_predict.py': combined,
            'model/stage2/s185_postprocess.py': runtime,
            'model/licenses/S185-NOTICE.txt': (HERE/'phase8/S185-NOTICE.txt').read_bytes(),
        }
        for name, content in new_sources.items():
            assert name not in expected
            target = root/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            expected[name] = sha(target)
        wrapper = '''\n\n# S185: exact S175, then crossing side/shift and S181 gap without S176.\n_S185_BASE_STAGE2 = predict_stage2\n\ndef predict_stage2(data_dir, model_dir):\n    import importlib.util\n    from pathlib import Path\n    base = _S185_BASE_STAGE2(data_dir, model_dir)\n    try:\n        source = Path(__file__).resolve().parent / 'model/stage2/s185_postprocess.py'\n        spec = importlib.util.spec_from_file_location('_s185_postprocess', source)\n        module = importlib.util.module_from_spec(spec)\n        spec.loader.exec_module(module)\n        return module.apply(globals(), base, data_dir, model_dir)\n    except Exception as exc:\n        globals()['_S185_DIAGNOSTICS'] = {'load_error': type(exc).__name__ + ': ' + str(exc), 'clips': {}}\n        return base\n'''
        inference = root/'inference.py'
        assert inference.read_bytes().endswith(b'        return base\n')
        inference.write_bytes(inference.read_bytes() + wrapper.encode())
        expected['inference.py'] = sha(inference)
        for path in root.rglob('*.py'):
            compile(path.read_bytes(), str(path), 'exec')
        report = dict(carrier=str(carrier), carrier_sha256=sha(carrier),
                      s174_source_sha256=hashlib.sha256(source).hexdigest(),
                      arms=[dict(path=str(p), sha256=pins[p.name]) for p in arms],
                      patches=[dict(path=str(p), sha256=patches[p.name]) for p in patch_paths],
                      dropped=['S176 first-frame bucket', 'S176 first-observation bucket',
                               'S178 side from S176 buckets'],
                      retained=['S178 accepted S161 crossing side', 'S180 S174-only +0.2s',
                                'S181 completed-S174 fallback gap'],
                      changed_from_carrier=['inference.py'],
                      added_to_carrier=sorted(new_sources), members=expected,
                      build_script_sha256=sha(__file__),
                      extension_sha256=sha(HERE/'phase8/s185_extension.py'),
                      postprocessor_sha256=sha(HERE/'phase8/s185_postprocess.py'))
        (root.parent/'build.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='members'}, indent=2), flush=True)

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
    parser.add_argument('--action', choices=['stage', 'overlay', 'overlay-bucket', 'overlay-chain', 'overlay-nobucket', 'package', 'all'], default='all')
    parser.add_argument('--carrier', help='Exact S172 carrier for the S171-relative S174 overlay')
    parser.add_argument('--normalize-merge-path', action='append', default=[],
                        help='Explicitly reviewed overlap whose CRLF is normalized before merging')
    args = parser.parse_args()
    if args.action in ('stage', 'all'):
        stage(args)
    if args.action == 'overlay':
        overlay(args)
    if args.action == 'overlay-bucket':
        overlay_bucket(args)
    if args.action == 'overlay-chain':
        overlay_chain(args)
    if args.action == 'overlay-nobucket':
        overlay_nobucket(args)
    if args.action in ('package', 'all'):
        package(args)
