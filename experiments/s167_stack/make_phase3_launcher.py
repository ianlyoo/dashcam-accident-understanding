"""Derive a phase-specific launcher without changing earlier evidence sources."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'launch_phase2b.ps1').read_text()
s=s.replace("'phase2b'","'phase3'").replace('Sol Worker S170','Sol Worker S171')
start=s.index('$jobs = ');end=s.index('\nforeach ($job',start)
s=s[:start]+"$jobs = @(@(2,'entry','../phase2b/candidate','qa_baseline'), @(2,'public','candidate','qa'), @(2,'long','candidate','qa'), @(2,'cascade','candidate','qa'), @(2,'entry','candidate','qa'), @(2,'fault','candidate','qa'), @(1,'public','candidate','qa'), @(1,'extras','candidate','qa'), @(3,'public','candidate','qa'))"+s[end:]
start=s.index('    $qaScript = ');end=s.index('    $name = ',start)
s=s[:start]+'''    $qaScript = 'qa_phase3.py'
    $ramCap = 4
    $extra = @()
    if ($outputName -eq 'qa_baseline') { $extra = @('--baseline') }
'''+s[end:]
s=s.replace("$report.s144_source_sha256 -eq $s144Hash", "$report.source_sha256.'model/stage2/s144/predict.py' -eq $s144Hash")
s=s.replace('$report.inference_sha256 -eq $entryHash', "$report.source_sha256.'inference.py' -eq $entryHash")
s=s.replace('$report.s162_source_sha256 -eq $s162Hash', "$report.source_sha256.'model/stage3/s141/predict.py' -eq $s162Hash")
s=s.replace('    while ($true) {', "    while ($true) {\n        if ((Get-Date) -ge [datetimeoffset]::Parse('2026-09-28T05:20:00+09:00').LocalDateTime) { throw 'S171 queue deadline reached' }")
s=s.replace("'--','/usr/bin/env'", "'--','/usr/bin/timeout','--signal=TERM','--kill-after=10s','1500s','/usr/bin/env'")
s=s.replace('        while (!$qaProcess.HasExited) {', '        $qaStarted = Get-Date\n        while (!$qaProcess.HasExited) {\n            if (((Get-Date)-$qaStarted).TotalSeconds -gt 1530) { throw "Bounded WSL command did not exit after timeout" }')
s=s.replace('$qaProcess.WaitForExit()', "if (!$qaProcess.WaitForExit(5000)) { throw 'QA output drain timed out' }")
s=s.replace('# Fresh S160 cascade reference, merged Stage2/fault, then Stage1 and Stage3 parity.', '# Pinned S170 diagnostic reference; exact S161 entry replay and unchanged-field QA.')
(HERE/'launch_phase3.ps1').write_text(s,encoding='utf-8')
