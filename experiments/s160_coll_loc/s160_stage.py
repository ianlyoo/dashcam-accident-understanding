"""Stage S160: add the S160 temporal localizer to the collision package of the S156_casc release.

Only model/stage2/s144/predict.py changes and s160_loc.py, localizer.json, loc_nn.npz, S160-NOTICE.txt are added;
every other member (Stage1, Stage3, entry/side/evasion code, rules, the S156 refiner) stays byte-identical to S156_casc.
Copied from candidates/s143_s144_overnight/s147_stage.py.
usage: python -B s160_stage.py S160 [--name S160_loc]
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from candidates.s118_s2rule import build  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = 'model/stage2/s144/'
MODIFIED = [PKG + 'predict.py']
ADDED = [PKG + 'S160-NOTICE.txt', PKG + 'loc_nn.npz', PKG + 'localizer.json', PKG + 's160_loc.py']
CONFIG = {'S160': ('S160_loc', 'S156_casc', 'e7fb70a6')}
PKGDIR = {'S160': 's160_pkg'}


def zip_members(path):
    with zipfile.ZipFile(path) as z:
        return {i.filename: {'bytes': i.file_size, 'sha256': __import__('hashlib').file_digest(z.open(i), 'sha256').hexdigest()}
                for i in z.infolist()}


def diff(old, new):
    modified = sorted(k for k in old if k in new and old[k] != new[k])
    added = sorted(set(new) - set(old)); removed = sorted(set(old) - set(new))
    assert modified == MODIFIED and added == ADDED and not removed, (modified, added, removed)
    return {'modified': modified, 'added': added, 'removed': removed, 'unchanged_members': len(old) - len(modified)}


def stage(cid):
    name, source, prefix = CONFIG[cid]
    stages = build.DATA / 'stage2_s118'
    archive = build.DATA / 'releases' / (source + '.zip')
    source_id = build.identity(archive); assert source_id['sha256'].startswith(prefix), source_id
    originals = zip_members(archive)
    old = build.read(stages / source / 'stage.json')
    assert originals == old['members'] == build.tree(stages / source / 'candidate')
    prior = build.read(stages / (source + '_qa') / 'qa.json')
    assert prior['passed'] and prior['candidate_manifest_sha256'] == build.manifest_sha(originals)
    assert build.identity(HERE.parent / 's143_s144_overnight/s156_pkg/predict.py') == old['members'][PKG + 'predict.py'], 'S156 predict source moved'
    dest = stages / name; dest.mkdir(exist_ok=False)
    root = dest / 'candidate'
    shutil.copytree(stages / source / 'candidate', root)
    pkgdir = HERE / PKGDIR[cid]
    for member in MODIFIED + ADDED:
        (root / member).write_bytes((pkgdir / member[len(PKG):]).read_bytes())
    members = build.tree(root)
    proof = diff(originals, members)
    receipt = dict(old, candidate_id=cid, root=str(root), members=members, manifest_sha256=build.manifest_sha(members),
                   comparison_base=source,
                   s160={'source_zip': str(archive), 'source_identity': source_id, 'source_qa_identity':
                         build.identity(stages / (source + '_qa') / 'qa.json'), 'identity_diff': proof})
    receipt['added_files'] = dict(old['added_files'])
    for member in MODIFIED + ADDED:
        receipt['added_files'][member] = {'from': str(pkgdir / member[len(PKG):]), **members[member]}
    receipt['new_model_weights'] = True
    build.save_new(dest / 'stage.json', receipt)
    print(json.dumps({'candidate': cid, 'stage': str(dest), 'manifest': receipt['manifest_sha256'], 'diff': proof}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 3 and sys.argv[2] == '--name':
        CONFIG['S160'] = (sys.argv[3],) + CONFIG['S160'][1:]
    stage(sys.argv[1])
