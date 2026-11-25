# 측정 전 정리 (운영 규칙 5절, docs/ENVIRONMENT.md 5절).
#
#   기록만 (기본):  powershell -NoProfile -ExecutionPolicy Bypass -File env\prep-measure.ps1
#   적용:           ... -File env\prep-measure.ps1 -Apply     (전원 구성표 '고성능' + 종료 목록의 앱 종료)
#   되돌리기:       ... -File env\prep-measure.ps1 -Restore   (적용 전 전원 구성표로)
#
# 기록은 env\prep-<YYYYMMDD-HHmm>-<before|after|restore>.txt 에 남는다. 처리량 실측 run 은
# 이 파일 이름을 원장 note 에 적어 같은 조건에서 쟀음을 남긴다.
# 종료 목록은 env\prep-stop-list.txt (한 줄에 프로세스 이름 하나, 확장자 없이). 처음에는 비어 있다.
# 기록에 나온 상주 프로그램을 보고 사람이 정해서 채운다. 안티치트·드라이버 서비스는 넣지 않는다.
param([switch]$Apply, [switch]$Restore)
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
Set-Location (Split-Path $PSScriptRoot -Parent)
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$stateFile = Join-Path $env:LOCALAPPDATA 'critconn-prev-powerscheme.txt'
$stopList = 'env\prep-stop-list.txt'

function Write-Snapshot([string]$tag) {
  $out = "env\prep-$stamp-$tag.txt"
  $lines = @()
  $lines += "# $tag $(Get-Date -Format o) $env:COMPUTERNAME"
  $lines += '== 전원 구성표'
  $lines += (powercfg /getactivescheme)
  $lines += '== GPU'
  $lines += (nvidia-smi --query-gpu=pstate,temperature.gpu,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw --format=csv)
  $lines += '== GPU 를 쓰는 프로세스'
  $lines += (nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv)
  $lines += '== CPU 시간 상위 15 (Windows 기본 프로세스 제외)'
  $lines += Get-Process |
    Where-Object { $_.Path -and $_.Path -notlike "$env:WINDIR\*" } |
    Sort-Object CPU -Descending | Select-Object -First 15 |
    ForEach-Object { '{0,-28} cpu={1,8:N1}s  ws={2,6:N0}MB  {3}' -f $_.ProcessName, $_.CPU, ($_.WorkingSet64 / 1MB), $_.Path }
  $lines += '== 메모리'
  $os = Get-CimInstance Win32_OperatingSystem
  $lines += ('사용 가능 {0:N1}GB / 전체 {1:N1}GB' -f ($os.FreePhysicalMemory / 1MB), ($os.TotalVisibleMemorySize / 1MB))
  $lines | Out-File -Encoding utf8 $out
  Write-Host "기록: $out"
}

if ($Restore) {
  if (Test-Path $stateFile) {
    $prev = (Get-Content $stateFile).Trim()
    powercfg /setactive $prev
    Remove-Item $stateFile
    Write-Host "전원 구성표 복원: $prev"
  } else { Write-Host '되돌릴 기록이 없다' }
  Write-Snapshot 'restore'
  exit 0
}

Write-Snapshot 'before'
if (-not $Apply) { Write-Host '기록만 했다. 적용하려면 -Apply'; exit 0 }

# 고성능 구성표 (SCHEME_MIN). 적용 전 구성표를 기억해 -Restore 로 되돌린다.
$active = ((powercfg /getactivescheme) -replace '.*GUID:\s*([0-9a-f-]+).*', '$1').Trim()
if (-not (Test-Path $stateFile)) { $active | Out-File -Encoding ascii $stateFile }
powercfg /setactive SCHEME_MIN

if (Test-Path $stopList) {
  foreach ($name in Get-Content $stopList | Where-Object { $_ -and -not $_.StartsWith('#') }) {
    $p = Get-Process -Name $name.Trim() -ErrorAction SilentlyContinue
    if ($p) { $p | Stop-Process -ErrorAction Continue; Write-Host "종료: $name" }
  }
}
Start-Sleep -Seconds 5
Write-Snapshot 'after'
