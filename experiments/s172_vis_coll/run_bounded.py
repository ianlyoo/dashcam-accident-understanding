"""Run noninteractive inspection commands with an explicit wall timeout."""
import subprocess
import sys

if __name__ == "__main__":
    command = bytes.fromhex(sys.argv[2]).decode("utf-8")
    result = subprocess.run(
        ["powershell.exe",
         "-NoProfile", "-NonInteractive", "-Command", command],
        timeout=float(sys.argv[1]), check=False,
    )
    raise SystemExit(result.returncode)
