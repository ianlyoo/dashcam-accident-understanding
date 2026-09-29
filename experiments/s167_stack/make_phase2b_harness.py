"""One-time derivation of the phase2b harness; preserves phase2a evidence source."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'qa_phase2a.py').read_text()
s=s.replace("ap.add_argument('--candidate', required=True)", "ap.add_argument('--reference-s160', action='store_true')\n    ap.add_argument('--candidate', required=True)")
s=s.replace("== 'long' else 's144", "in ('long', 'fault') else 's144")
s=s.replace("if args.panel == 'fault':\n        import pandas as pd", "if args.panel == 'fault' and args.stage == 3:\n        import pandas as pd")
needle="    else:\n        frame = ns['predict_stage%d' % args.stage](inputs, root / ('model/stage%d' % args.stage))"
replacement='''    elif args.stage == 2 and args.panel == 'fault':
        import pandas as pd
        expected = pd.read_csv(data / 's160_coll_loc/qa/candidate_long_none.csv', dtype={'ID':str})
        baseline = pd.read_csv(data / 's164_runtime/qa_base/stage2_long.csv', dtype={'ID':str})
        fault_clip = '060'  # second clip is worker-prefetched; inject 019 as well for observable change.
        fault_ids = {'019', '060'}
        package = ns['_s118_adapter']().load_package(root, 'model/stage2/s144')
        original_init, original_locate = package.Reranker.__init__, package.Reranker.locate
        emb = ns['_S2_COLLISION_NAMESPACE']
        original_hooks = (emb['extract_folder_features'], emb['locate_collision'], emb['predict_folder'],
                          cv2.imread, ns['_S118_BASE_PREDICT_STAGE2'])
        active, injected = [None], []
        def init(self, *a, **kw):
            original_init(self, *a, **kw)
            assert self.loc is not None and self.loc_error is None
            actual = self.L.localize
            def fail(*a, **kw):
                if active[0] in fault_ids:
                    injected.append(active[0])
                    raise RuntimeError('S170 injected localizer failure')
                return actual(*a, **kw)
            self.L.localize = fail
        def locate(self, features, paths):
            active[0] = paths[0].parent.name
            try:
                return original_locate(self, features, paths)
            finally:
                active[0] = None
        package.Reranker.__init__, package.Reranker.locate = init, locate
        try:
            faulted = ns['predict_stage2'](inputs, root / 'model/stage2')
        finally:
            package.Reranker.__init__, package.Reranker.locate = original_init, original_locate
        def normalized(f):
            f = f.copy(); f.ID = f.ID.astype(str)
            return f.reset_index(drop=True)
        want = expected.copy()
        for ident in fault_ids:
            want.loc[want.ID == ident, :] = baseline.loc[baseline.ID == ident, :].values
        assert normalized(faulted).equals(want), 'Localizer fail-closed parity'
        assert sorted(injected) == sorted(fault_ids)
        assert int(faulted.loc[faulted.ID == '019', 'collision_frame'].iloc[0]) == 612
        assert int(expected.loc[expected.ID == '019', 'collision_frame'].iloc[0]) == 609
        assert int(faulted.loc[faulted.ID == '076', 'collision_frame'].iloc[0]) == 597
        fault_diag = dict(ns['_S164_LAST_DIAGNOSTICS'])
        fault_s118 = dict(ns['_S118_LAST_DIAGNOSTICS'])
        assert fault_diag['prefetched'] == 3 and fault_diag['unused_features'] == 0
        assert not fault_diag.get('errors') and not fault_diag.get('lazy_scene_error')
        for ident in fault_ids:
            d = fault_s118['s142_clips'][ident]
            assert d.get('loc_error') and 'loc_to' not in d
        def restored():
            return original_hooks == (emb['extract_folder_features'], emb['locate_collision'], emb['predict_folder'],
                                      cv2.imread, ns['_S118_BASE_PREDICT_STAGE2'])
        assert restored()
        fault_bytes = faulted.to_csv(index=False, lineterminator='\\n').encode()
        (output / 'stage2_fault_injected.csv').write_bytes(fault_bytes)
        gc.collect(); torch.cuda.empty_cache()
        frame = ns['predict_stage2'](inputs, root / 'model/stage2')
        assert normalized(frame).equals(expected) and restored()
        fault_report = dict(injected_ids=injected, prefetched_fault_clip=fault_clip,
            observable_fallback_clip='019', s156_collision=612, s160_collision=609,
            unaffected_clip='076', unaffected_collision=597, next_call_recovers_s160=True,
            all_hooks_restored=True, fault_s164=fault_diag, fault_s118=fault_s118,
            fault_csv_sha256=hashlib.sha256(fault_bytes).hexdigest())
    else:
        frame = ns['predict_stage%d' % args.stage](inputs, root / ('model/stage%d' % args.stage))'''
assert needle in s
s=s.replace(needle,replacement)
s=s.replace("report.update(resources=resources,", "report.update(reference_s160=args.reference_s160,\n                  s144_source_sha256=hashlib.sha256((root / 'model/stage2/s144/predict.py').read_bytes()).hexdigest(),\n                  build_sha256=hashlib.sha256((output.parent / 'build.json').read_bytes()).hexdigest(),\n                  resources=resources,")
s=s.replace("    if args.stage == 3:\n        assert not any", """    if args.stage == 2:
        import pandas as pd
        baseline = pd.read_csv(data / 's164_runtime/qa_base' / ('stage2_' + ('long' if args.panel == 'fault' else args.panel) + '.csv'), dtype={'ID':str})
        actual = frame.copy(); actual.ID = actual.ID.astype(str)
        protected = ['ID','entry_side','evasion_space']
        assert actual[protected].reset_index(drop=True).equals(baseline[protected])
        assert all(int(e) <= int(c) for e,c in zip(actual.entry_frame, actual.collision_frame))
        report['protected_fields_equal_s156'] = True
        clips = report['s118'].get('s142_clips', {})
        if args.panel != 'public': assert len(clips) == len(frame)
        for detail in clips.values():
            assert not any(detail.get(k) for k in ('refine_error','loc_error','loc_load_error'))
            assert 'loc_to' in detail
            if not args.reference_s160:
                assert detail['detector_batch'] == int(os.environ.get('S164_DETECT_BATCH','1'))
        if not args.reference_s160:
            assert report['s164']['prefetched'] == len(frame)
            assert 'lazy_scene_skipped' in report['s164']
    if args.stage == 3:
        assert not any""")
s=s.replace("        reference = data / 's164_runtime/qa_base' / (label + '.csv')", """        if args.reference_s160:
            assert args.stage == 2 and args.panel == 'cascade'
            reference = output / (label + '.csv')
            report['reference_generation_only'] = True
        elif args.panel == 'cascade':
            reference = output.parent / 'qa_reference/stage2_cascade.csv'
        else:
            reference = data / 's160_coll_loc/qa' / ('candidate_' + ('long' if args.panel == 'fault' else args.panel) + '_none.csv')""")
s=s.replace("if args.panel == 'fault':\n        expected = pd.read_csv(reference)", "if args.panel == 'fault' and args.stage == 3:\n        expected = pd.read_csv(reference)")
for short, full in [('019','00019'),('060','00060'),('076','00076')]:
    s=s.replace("'"+short+"'", "'"+full+"'")
s=s.replace('1.75 * 1024**3', '1.65 * 1024**3')
s=s.replace("    times = {}", """    if args.stage == 2:
        # Release allocator slack before loading reranker/localizer. Do not wrap
        # _S118_BASE_PREDICT_STAGE2: S164 inspects its code for lazy scene support.
        package = ns['_s118_adapter']().load_package(root, 'model/stage2/s144')
        original_package_init = package.Reranker.__init__
        def trim_init(self, *a, **kw):
            gc.collect()
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)
            return original_package_init(self, *a, **kw)
        package.Reranker.__init__ = trim_init
    times = {}""")
# All candidates are fully hashed in the launcher and again before packaging.
(HERE/'qa_phase2b.py').write_text(s, encoding='utf-8')

s=(HERE/'launch_phase2a.ps1').read_text().replace("'phase2a'", "'phase2b'").replace('qa_phase2a.py','qa_phase2b.py').replace('Sol Worker S169','Sol Worker S170')
s=s.replace("$jobs = @(@(3,'public'), @(3,'fault'), @(1,'public'), @(1,'extras'), @(2,'public'), @(2,'long'), @(2,'cascade'))", """$jobs = @(@(2,'cascade','reference_s160','qa_reference'), @(2,'public','candidate','qa'), @(2,'long','candidate','qa'), @(2,'cascade','candidate','qa'), @(2,'fault','candidate','qa'), @(1,'public','candidate','qa'), @(1,'extras','candidate','qa'), @(3,'public','candidate','qa'))""")
s=s.replace('    $stage, $panel = $job', '    $stage, $panel, $treeName, $outputName = $job\n    $extra = @()\n    if ($treeName -eq "reference_s160") { $extra = @("--reference-s160") }')
s=s.replace('$work/candidate/', '$work/$treeName/')
s=s.replace('        if ($report.validation_ok', '''        $s144Hash = (Get-FileHash "$work/$treeName/model/stage2/s144/predict.py" -Algorithm SHA256).Hash.ToLower()
        $buildHash = (Get-FileHash "$work/build.json" -Algorithm SHA256).Hash.ToLower()
        if ($report.s144_source_sha256 -eq $s144Hash -and $report.build_sha256 -eq $buildHash -and $report.validation_ok''')
s=s.replace('S164_WORKERS=2 S164_DETECT_BATCH=1', 'MALLOC_ARENA_MAX=2 S164_WORKERS=2 S164_DETECT_BATCH=1')
s=s.replace('/$Phase/candidate"', '/$Phase/$treeName"')
s=s.replace('--stage $stage --panel $panel >', '--stage $stage --panel $panel @extra >')
s=s.replace('($totalCaps + 2) -le 16', '($totalCaps + 2) -le 14')
s=s.replace('# Changed Stage3 path first, then the unchanged-stage integration panels.', '# Fresh S160 cascade reference, merged Stage2/fault, then Stage1 and Stage3 parity.')
(HERE/'launch_phase2b.ps1').write_text(s,encoding='utf-8')
