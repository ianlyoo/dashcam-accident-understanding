"""Hard timeout for one noninteractive subprocess, with bounded log output."""
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--seconds',type=int,required=True);p.add_argument('--tail',type=int,default=5000);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
try:
 r=subprocess.run(a.command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=a.seconds)
 sys.stdout.buffer.write(r.stdout[-a.tail:]);sys.stderr.buffer.write(r.stderr[-a.tail:]);sys.exit(r.returncode)
except subprocess.TimeoutExpired as e:
 sys.stderr.write('TIMEOUT '+str(a.seconds)+'s\n');sys.exit(124)
