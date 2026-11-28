# 본체에서 대기열을 SSH 세션과 분리해 띄운다 (TROUBLESHOOTING T-013).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 configs\queue\p2.yaml
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\launch_detached.ps1 configs\queue\p2.yaml -DryRun
#
# - WMI Win32_Process.Create 로 띄운다. Start-Process 로 띄운 프로세스는 SSH 명령이 끝나면 같이 죽고,
#   작업 스케줄러 1회 작업은 대화형 로그온이 없으면 실행되지 않았다 (T-013, 2026-11-28 실측).
# - 화면 출력은 experiments/logs/launch-<대기열>-<시각>.log 에 남는다. run_queue.py 도 같은 폴더에
#   자기 로그를 쓴다. 띄운 뒤 SSH 를 끊어도 된다.
# - 진행 확인: experiments/LEDGER.tsv 마지막 행 · 로그 끝 · Get-Process -Id <PID>.
# - 같은 대기열이 이미 돌고 있으면 띄우지 않는다.
param(
    [Parameter(Mandatory = $true)][string]$Queue,
    [switch]$DryRun,
    [string]$Python = "C:\ProgramData\anaconda3\envs\critconn\python.exe"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$QueuePath = (Resolve-Path (Join-Path $Root $Queue)).Path
$Name = [IO.Path]::GetFileNameWithoutExtension($QueuePath)

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like "*run_queue.py*$Name*" }
if ($running) {
    Write-Output "이미 도는 대기열이 있다: PID $($running.ProcessId -join ', ')"
    exit 1
}

$Logs = Join-Path $Root "experiments\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd-HHmm"
$Log = Join-Path $Logs "launch-$Name-$Stamp.log"
$Extra = if ($DryRun) { " --dry-run" } else { "" }
# cmd 리디렉션은 바이트를 그대로 쓴다 (PowerShell 5.1 의 *> 는 UTF-16 으로 쓴다, T-005)
$Cmd = "cmd /c `"set PYTHONIOENCODING=utf-8&& `"$Python`" -u scripts\run_queue.py `"$QueuePath`"$Extra > `"$Log`" 2>&1`""

$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine      = $Cmd
    CurrentDirectory = $Root
}
if ($r.ReturnValue -ne 0) {
    Write-Output "띄우기 실패: Win32_Process.Create 반환 $($r.ReturnValue)"
    exit 1
}
Write-Output "띄움: PID $($r.ProcessId) (cmd), 로그 $Log"
