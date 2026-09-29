"""Detached bounded CPU job with S174-only logs."""
import argparse,datetime,json,pathlib,subprocess,sys
H=pathlib.Path(__file__).resolve().parent;D=pathlib.Path('$DATA_DIR/s174_cov');(D/'logs').mkdir(parents=True,exist_ok=True)
p=argparse.ArgumentParser();p.add_argument('script');p.add_argument('--seconds',type=int,default=7200);a=p.parse_args()
assert a.script in ('anchors.py','audit_labeled.py','audit_750.py','eval_candidate.py','stage.py','finalize.py')
prefix=D/'logs'/(a.script[:-3]+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
with open(str(prefix)+'.out.log','wb') as out,open(str(prefix)+'.err.log','wb') as err:
 proc=subprocess.Popen([sys.executable,'-B',str(H/'run_bounded.py'),'--seconds',str(a.seconds),'--tail','1000000',sys.executable,'-B',str(H/a.script)],
  cwd=H.parents[1],stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.DETACHED_PROCESS)
print(json.dumps(dict(pid=proc.pid,log=str(prefix))))
