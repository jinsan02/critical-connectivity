# 커밋 전 전체 검증 — 환경 → 린트 → 포맷 → 테스트 → 원장 → push 안 한 커밋의 메시지
# 사용: powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
$py = 'C:\ProgramData\anaconda3\envs\critconn\python.exe'
if (-not (Test-Path $py)) { $py = Join-Path $env:USERPROFILE 'anaconda3\envs\critconn\python.exe' }
if (-not (Test-Path $py)) { $py = 'python' }
# conda-forge 의 ruff 는 python 모듈이 아니라 Library\bin\ruff.exe 다
$ruff = Join-Path (Split-Path $py -Parent) 'Library\bin\ruff.exe'
if (Test-Path $ruff) { $ruffExe = $ruff; $ruffPre = @() } else { $ruffExe = $py; $ruffPre = @('-m', 'ruff') }
Write-Host "python: $py"
Write-Host "ruff  : $ruffExe $($ruffPre -join ' ')"
$fail = 0

Write-Host '== 환경 (numpy 행렬곱)'
# pytest 는 conftest 가 MKL 계층을 정해 이 문제를 가린다. 여기서는 사용자 환경 그대로 본다 (T-012)
& $py -c "import numpy as np; a = np.ones((64, 64)); assert (a @ a)[0, 0] == 64"
if ($LASTEXITCODE -ne 0) {
  $fail = 1
  Write-Host '   실패 — MKL_THREADING_LAYER=SEQUENTIAL 사용자 환경 변수를 확인 (TROUBLESHOOTING T-012)'
} else { Write-Host "   통과 (MKL_THREADING_LAYER=$env:MKL_THREADING_LAYER)" }

Write-Host '== 린트 (ruff check)'
& $ruffExe @ruffPre check .
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host '== 포맷 (ruff format --check)'
& $ruffExe @ruffPre format --check .
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host '== 테스트'
& $py -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { $fail = 1 }

Write-Host '== 원장 무결성'
& $py tools\validate_ledger.py
if ($LASTEXITCODE -ne 0) { $fail = 1 } else { Write-Host '   통과' }

Write-Host '== push 안 한 커밋의 메시지'
$upstream = git rev-parse --abbrev-ref '@{u}' 2>$null
# upstream 이 없는 새 브랜치는 origin/main 과 비교한다 (전체 이력을 다시 보지 않는다)
$base = if ($upstream) { $upstream } elseif (git rev-parse -q --verify origin/main 2>$null) { 'origin/main' } else { $null }
$range = if ($base) { "$base..HEAD" } else { 'HEAD' }
$shas = git rev-list $range 2>$null
if (-not $shas) { Write-Host '   새 커밋 없음' }
foreach ($sha in $shas) {
  Write-Host "   $(git log -1 --format='%h' $sha)"
  # 메시지는 검사기가 git 에서 바로 읽는다 (PowerShell 파이프는 한국어를 깨뜨린다)
  & $py tools\check_commit_msg.py --commit $sha
  if ($LASTEXITCODE -ne 0) { $fail = 1 }
}

if ($fail) { Write-Host "`n검증 실패" -ForegroundColor Red; exit 1 }
Write-Host "`n검증 통과" -ForegroundColor Green
