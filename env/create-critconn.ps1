# critconn 환경을 env/critconn.yml 대로 만든다 (docs/ENVIRONMENT.md 2절).
#
# conda env create 는 채널 덮어쓰기 옵션이 없어서, 본체 conda 설정의 defaults 채널
# (이용약관 동의 필요)을 피할 수 없다. 그래서 yml 을 읽어 conda create
# --override-channels -c conda-forge 로 만들고, 환경 .condarc 를 conda-forge 로 고정한 뒤
# pip 패키지를 설치한다. 패키지 목록의 출처는 yml 하나다.
#
# 사용: powershell -NoProfile -ExecutionPolicy Bypass -File env\create-critconn.ps1
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
$conda = 'C:\ProgramData\anaconda3\Scripts\conda.exe'
$envDir = 'C:\ProgramData\anaconda3\envs\critconn'

$condaPkgs = @(); $pipArgs = @(); $inDeps = $false; $inPip = $false
foreach ($line in Get-Content env\critconn.yml) {
  $l = ($line -replace '#.*$', '').TrimEnd()
  if ($l -match '^dependencies:') { $inDeps = $true; continue }
  if (-not $inDeps -or $l -eq '') { continue }
  if ($l -match '^\s{2}- pip:$') { $inPip = $true; continue }
  if ($inPip -and $l -match '^\s{6}- (.+)$') { $pipArgs += ($Matches[1] -split '\s+'); continue }
  if ($l -match '^\s{2}- (\S+)$') { $inPip = $false; $condaPkgs += $Matches[1] }
}
Write-Host "conda: $($condaPkgs -join ' ')"
Write-Host "pip  : $($pipArgs -join ' ')"

if (Test-Path "$envDir\python.exe") { throw "이미 있다: $envDir (지우려면 사람에게 묻는다)" }
& $conda create -y -n critconn --override-channels -c conda-forge @condaPkgs
if ($LASTEXITCODE -ne 0) { throw "conda create 실패 ($LASTEXITCODE)" }

"channels:`n  - conda-forge`nchannel_priority: strict`n" | Out-File -Encoding ascii "$envDir\.condarc"

& "$envDir\python.exe" -m pip install --no-input @pipArgs
if ($LASTEXITCODE -ne 0) { throw "pip install 실패 ($LASTEXITCODE)" }
& "$envDir\python.exe" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
