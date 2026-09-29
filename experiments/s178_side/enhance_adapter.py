# S178: consume only the S161/S176 side decision from the tracker diagnostics.
# The S118 rule keeps side=false because its collision-preservation validator
# requires it. This wrapper updates side after the exact S176 track call.
try:
    _s178_previous_predict_track = _predict_track

    def _predict_track(ns, data_dir, model_dir, rule, base_call=None, diagnostics=None):
        out = _s178_previous_predict_track(ns, data_dir, model_dir, rule,
                                           base_call=base_call, diagnostics=diagnostics)
        try:
            if rule.get('entry_frame', {}).get('package') != 'model/stage2/s161':
                return out
            records = ns.get('_S118_LAST_DIAGNOSTICS', {}).get('clips', {})
            if not records or 'entry_side' not in out:
                return out
            revised = out.copy()
            for index, row in out.iterrows():
                record = records.get(str(row['ID']), {})
                if record.get('error'):
                    continue
                result = record.get('result') or {}
                if result.get('reason') not in ('crossing', 's176_inside_from_start',
                                                's176_inside_first_minus_0p2s'):
                    continue
                side = result.get('side')
                if side in ('LEFT', 'RIGHT'):
                    revised.at[index, 'entry_side'] = side
            return revised
        except Exception:
            # Any side-application fault returns the exact S176 output.
            return out
except Exception:
    pass
