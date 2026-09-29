"""Best-effort atomic telemetry; missing samples never terminate candidate QA."""
import argparse
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream)
            stream.flush()
        for attempt in range(6):  # initial attempt plus five retries
            try:
                os.replace(temporary, path)
                return True
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.5)
    except Exception as exc:
        print(f'WARNING: skipped monitor sample {path.name}: {exc}', flush=True)
        return False
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def monitor_resources(output, label):
    admission = float(os.environ['S171_ADMISSION_FREE_COMMIT_GIB'])
    state = dict(rss_cap_gib=4, minimum_host_free_commit_gib=admission,
                 admission_free_commit_gib=admission,
                 peak_aggregate_rss_bytes=0, peak_processes=0, sample_interval_seconds=0.25)
    stop = threading.Event()

    def sample():
        last_warning = 0
        while not stop.is_set():
            pending, seen, rss = [os.getpid()], set(), 0
            while pending:
                pid = pending.pop()
                if pid in seen:
                    continue
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                    rss += next((int(line.split()[1]) * 1024 for line in status.splitlines() if line.startswith('VmRSS:')), 0)
                    seen.add(pid)
                    for task in Path(f'/proc/{pid}/task').iterdir():
                        pending.extend(int(n) for n in (task / 'children').read_text().split())
                except (OSError, ValueError):
                    continue
            state['peak_aggregate_rss_bytes'] = max(rss, state['peak_aggregate_rss_bytes'])
            state['peak_processes'] = max(len(seen), state['peak_processes'])
            low_commit = False
            try:
                commit_path = output.parent / 'host_commit.json'
                commit = json.loads(commit_path.read_text())
                if time.time() - float(commit['sampled_epoch']) > 30:
                    raise ValueError('Host commit monitor stale; retaining prior samples')
                free = float(commit['free_commit_gib'])
                previous = state['minimum_host_free_commit_gib']
                state['minimum_host_free_commit_gib'] = free if previous is None else min(previous, free)
                low_commit = free < 12
            except Exception as exc:
                state['host_commit_skipped_reads'] = state.get('host_commit_skipped_reads', 0) + 1
                if time.monotonic() - last_warning >= 30:
                    print(f'WARNING: host commit sample unavailable: {exc}', flush=True)
                    last_warning = time.monotonic()
                    state['host_commit_warnings'] = state.get('host_commit_warnings', 0) + 1
            # Actual observed resource overruns still enforce the sharing cap.
            if rss > 4 * 2**30 or len(seen) > 2 or low_commit:
                state['resource_cap_exceeded'] = True
                atomic_json(output / (label + '_resource_failure.json'), state)
                for pid in seen - {os.getpid()}:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                os._exit(80)
            stop.wait(0.25)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return state, stop, thread


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--free', type=float, required=True)
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    atomic_json(args.path, dict(free_commit_gib=args.free, sampled_epoch=time.time(), job=args.job))
