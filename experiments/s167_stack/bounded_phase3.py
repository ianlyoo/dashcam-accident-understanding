"""CPU child commands always have timeouts; never launch another agent."""
import argparse
import subprocess
import sys
ap=argparse.ArgumentParser()
ap.add_argument('--seconds',type=float,required=True)
ap.add_argument('command',nargs=argparse.REMAINDER)
a=ap.parse_args()
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
try:
    result=subprocess.run(cmd,timeout=a.seconds,check=False)
except subprocess.TimeoutExpired:
    print('COMMAND TIMEOUT',a.seconds,flush=True)
    sys.exit(124)
sys.exit(result.returncode)
