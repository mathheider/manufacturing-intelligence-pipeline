# One-shot reproduce (Windows PowerShell): rebuild data, benchmark, tests.
# No third-party packages required.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONPATH = "src"

Write-Host "== 1/3  rebuild frozen data =="
python scripts/build_data.py

Write-Host "== 2/3  canonical benchmark (14 comps, 40 seeds) =="
python -m mip.cli bench --seeds 40

Write-Host "== 3/3  test suite (incl. golden-number regression) =="
python -m pytest -q
