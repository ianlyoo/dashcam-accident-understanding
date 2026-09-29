"""Bounded inspection of explicitly named task-owned process trees."""
import json
import sys
import psutil
if __name__=='__main__':
    rows=[]
    for value in sys.argv[1:]:
        try:
            root=psutil.Process(int(value)); procs=[root]+root.children(recursive=True)
        except psutil.NoSuchProcess: continue
        for proc in procs:
            try: rows.append(dict(pid=proc.pid,parent=proc.ppid(),rss_gib=proc.memory_info().rss/2**30))
            except psutil.NoSuchProcess: pass
    print(json.dumps(dict(processes=rows,total_rss_gib=sum(r['rss_gib'] for r in rows))),flush=True)
