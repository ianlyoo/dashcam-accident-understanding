"""CPU regression checks for the specific Windows telemetry failures."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from candidates.s167_stack import monitor_phase3 as monitor

with tempfile.TemporaryDirectory(prefix='s171_monitor_') as directory:
    root = Path(directory)
    target = root / 'host_commit.json'
    real_replace = os.replace
    attempts = []
    def transient(source, destination):
        attempts.append(source)
        if len(attempts) < 3:
            raise PermissionError('injected transient sharing violation')
        return real_replace(source, destination)
    with patch.object(monitor.os, 'replace', side_effect=transient), patch.object(monitor.time, 'sleep') as sleep:
        assert monitor.atomic_json(target, {'value': 42})
        assert len(attempts) == 3 and sleep.call_count == 2
    assert json.loads(target.read_text()) == {'value': 42}
    with patch.object(monitor.os, 'replace', side_effect=PermissionError('persistent test')), patch.object(monitor.time, 'sleep') as sleep:
        assert not monitor.atomic_json(target, {'value': 99})
        assert sleep.call_count == 5
    assert json.loads(target.read_text()) == {'value': 42}
    assert not list(root.glob('*.tmp'))
    output = root / 'qa'
    output.mkdir()
    monitor.atomic_json(target, {'sampled_epoch': 0, 'free_commit_gib': 1})
    with patch.dict(os.environ, S171_ADMISSION_FREE_COMMIT_GIB='64'), patch.object(monitor.os, '_exit') as exit_call:
        state, stop, thread = monitor.monitor_resources(output, 'test')
        time.sleep(0.6)
        stop.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and not exit_call.called
        assert state['host_commit_warnings'] >= 1
        assert state['minimum_host_free_commit_gib'] == 64
print('PASS: atomic retry, exhausted retries skip, prior JSON preserved, temp cleanup, stale warning without abort')
