"""run_id 이름 규칙과 원장 쓰기 (STRUCTURE.md 5절, docs/LEDGER_SCHEMA.md v1, 2026-11-27 확정).

run_id = ``<phase>_<exp>_<arch>_<변형>_i<init>m<mask>o<order>``  예: ``p2_exp02_cross_r0.04_i1m2o0``

- phase: ``p0``~``p7`` · ``ea``~``ec`` (§15.1, §22.1-8)
- exp: ``exp01``~``exp11`` · ``r1`` · ``r2`` · ``r_rfm``
- arch: 모델 설정 이름의 짧은 이름 (``ARCH_CODES``)
- 변형: 연결 밀도가 있는 모델은 ``r<밀도>``, 아니면 ``base``. 학습률 등 다른 변형은 손으로 준다
- mask: ``mask_seed`` 가 없는 모델은 ``x``

원장 한 행 = 실행 하나의 상태 변화 하나. 학습 코드가 시작에 ``start``, 끝에 ``ok`` 또는
``fail`` 을 쓴다.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from tools.validate_ledger import COLUMNS, RUN_ID_RE

ARCH_CODES = {
    "r1": "r1",
    "dense_8m": "dense8m",
    "dense_64m": "dense64m",
    "dense_64m_8l": "dense64m8l",
    "ensemble_8x8m": "ens8x8m",
    "parallel_mixing": "pmix",
    "cross_model": "cross",
    "cross_model_hash": "crosshash",
    "cross_model_random": "crossrand",
}
PHASE_RE = re.compile(r"^(p[0-7]|e[a-c])$")
EXP_RE = re.compile(r"^(exp(0[1-9]|1[01])|r1|r2|r_rfm)$")
DENSITY_KINDS = ("parallel_mixing", "cross_model")


def arch_code(model_cfg: dict) -> str:
    name = model_cfg["name"]
    if name not in ARCH_CODES:
        raise ValueError(f"짧은 이름이 없는 모델: {name!r} — ledger.ARCH_CODES 에 더한다")
    return ARCH_CODES[name]


def default_variant(model_cfg: dict) -> str:
    if model_cfg.get("kind") in DENSITY_KINDS:
        return f"r{float(model_cfg.get('density', 1.0)):g}"
    return "base"


def make_run_id(
    phase: str, exp: str, model_cfg: dict, train_cfg: dict, variant: str | None = None
) -> str:
    if not PHASE_RE.match(phase):
        raise ValueError(f"phase 는 p0~p7 · ea~ec: {phase!r}")
    if not EXP_RE.match(exp):
        raise ValueError(f"exp 는 exp01~exp11 · r1 · r2 · r_rfm: {exp!r}")
    mask = model_cfg.get("mask_seed") if model_cfg.get("kind") in DENSITY_KINDS else None
    rid = (
        f"{phase}_{exp}_{arch_code(model_cfg)}_{variant or default_variant(model_cfg)}"
        f"_i{train_cfg.get('init_seed', 0)}m{'x' if mask is None else mask}"
        f"o{train_cfg.get('order_seed', 0)}"
    )
    if not RUN_ID_RE.match(rid):
        raise ValueError(f"run_id 형식이 틀렸다: {rid!r}")
    return rid


def _cell(v) -> str:
    if v is None:
        return "NA"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("\t", " ").replace("\n", " ")


def ledger_row(fields: dict, status: str, note: str = "") -> str:
    """원장 한 줄 (탭 구분, 끝 줄바꿈 없음). 없는 열은 NA."""
    row = dict(fields)
    row["status"] = status
    row["note"] = note
    row["ts_utc"] = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return "\t".join(_cell(row.get(c)) for c in COLUMNS)


def append(path: Path, line: str) -> None:
    """원장에 한 줄을 덧붙인다. 파일이 없으면 헤더부터 만든다 (저장소 밖 시험 실행용)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\t".join(COLUMNS) + "\n", encoding="utf-8", newline="\n")
    header = path.read_text(encoding="utf-8").split("\n", 1)[0]
    if tuple(header.split("\t")) != COLUMNS:
        raise ValueError(f"{path}: 원장 헤더가 스키마와 다르다 (docs/LEDGER_SCHEMA.md)")
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")
