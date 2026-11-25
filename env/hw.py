"""본체 하드웨어 · 환경 실측 — 결과를 마크다운으로 출력한다.

사용 (본체):
    C:\\ProgramData\\anaconda3\\envs\\critconn\\python.exe env\\hw.py > env\\hw-YYYYMMDD.md
"""

from __future__ import annotations

import datetime as dt
import platform
import shutil
import subprocess
import sys
import time

import torch


def sh(cmd: str) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    return out.stdout.strip()


def ps(expr: str) -> str:
    return sh(f'powershell -NoProfile -Command "{expr}"')


def bf16_tflops(n: int = 4096, reps: int = 20) -> float | None:
    """참고값. 처리량 판단은 Gate 1 의 1K 스텝 벤치마크가 한다."""
    if not torch.cuda.is_available():
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * reps / (time.perf_counter() - t) / 1e12


def main() -> None:
    gpu = sh(
        "nvidia-smi --query-gpu=name,memory.total,driver_version,power.limit --format=csv,noheader"
    )
    cpu = ps("(Get-CimInstance Win32_Processor).Name")
    ram = ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)")
    osv = ps("(Get-CimInstance Win32_OperatingSystem).Caption")
    du = shutil.disk_usage("C:\\")
    ok = torch.cuda.is_available()
    cap = torch.cuda.get_device_capability(0) if ok else None
    free, total = torch.cuda.mem_get_info() if ok else (0, 0)
    tf = bf16_tflops()
    today = dt.date.today().isoformat()

    rows = [
        ("OS", f"{osv} ({platform.version()})"),
        ("CPU", cpu),
        ("RAM", f"{ram} GB"),
        ("GPU", gpu),
        (
            "C: 디스크",
            f"전체 {du.total / 1e9:,.0f}GB · 남음 {du.free / 1e9:,.0f}GB"
            " (기획서 §15.2 필요량 약 500GB)",
        ),
        ("Python", f"{sys.version.split()[0]} (`{sys.prefix}`)"),
        (
            "PyTorch",
            f"{torch.__version__} · CUDA {torch.version.cuda}"
            f" · cuDNN {torch.backends.cudnn.version()}",
        ),
        ("CUDA 사용 가능", f"{ok} · compute capability {cap}"),
        ("VRAM (torch)", f"남음 {free / 2**30:.1f} GiB / 전체 {total / 2**30:.1f} GiB"),
        (
            "BF16 matmul 4096²",
            f"{tf:.1f} TFLOPS (참고값)" if tf else "NA",
        ),
    ]
    print(f"# 본체 하드웨어 실측 — {today}\n")
    print("`env/hw.py` 로 본체(desktop-5070)에서 잰 값이다. 바뀌면 새 파일을 만든다.\n")
    print("| 항목 | 값 |\n| --- | --- |")
    for k, v in rows:
        print(f"| {k} | {v} |")


if __name__ == "__main__":
    main()
