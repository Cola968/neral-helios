param([string]$Data = 'curriculum', [int]$Steps = 160)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$neralPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $neralPython)) { throw 'Lokale Python-Umgebung fehlt. Siehe README.md.' }
$neralRun = 'runs/continue_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
Write-Host 'Der bisherige Pilot ueberangepasst nach Schritt 80. Vor einem neuen Lauf bessere Daten bereitstellen.'
& $neralPython -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests fehlgeschlagen; kein Training gestartet.' }
& $neralPython -u experiment.py --init-from seed/neral20m.pt --data $Data --out $neralRun --steps $Steps --lr 3e-5 --batch-size 8 --grad-accum 4 --eval-every 20 --patience 4
if ($LASTEXITCODE -ne 0) { throw 'Trainingslauf fehlgeschlagen. Protokoll und letzten Checkpoint pruefen.' }
Write-Host ('Lauf gespeichert unter ' + $neralRun + '. Ergebnisse pruefen, bevor ein neuer Checkpoint eingesetzt wird.')
