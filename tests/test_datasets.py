"""Gate 1 데이터 점검 — 분할 무결성 · 비율 · 해시 재현 · 숫자 빈도 (기획서 v0.5 §12B-7)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.datasets import generate, make_split, operand_frequency

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "data").glob("*.yaml"))
MAIN = {"train": 0.4, "val": 0.1, "calib": 0.1, "test": 0.4}


def test_generate_enumerates_all_pairs_with_correct_labels():
    d = generate(97)
    assert d.n == 97 * 97
    assert len({(int(a), int(b)) for a, b in zip(d.a, d.b, strict=True)}) == d.n
    assert np.array_equal(d.y, (d.a + d.b) % 97)
    t = d.tokens()
    assert t.shape == (d.n, 3) and (t[:, 2] == 97).all() and d.vocab_size == 98


def test_generate_rejects_unknown_op():
    with pytest.raises(ValueError):
        generate(97, "mul")


@pytest.mark.parametrize("method", ["pair", "commutative"])
def test_split_is_disjoint_and_complete(method):
    d = generate(97)
    s = make_split(d, method, 0, MAIN)
    allidx = np.concatenate(list(s.indices.values()))
    assert len(allidx) == d.n and len(np.unique(allidx)) == d.n


@pytest.mark.parametrize("method", ["pair", "commutative"])
def test_split_ratios_within_half_percent_point(method):
    d = generate(97)
    sizes = make_split(d, method, 0, MAIN).sizes()
    for name, frac in MAIN.items():
        assert abs(sizes[name] / d.n - frac) <= 0.005, (name, sizes[name])


def test_pair_split_matches_plan_table():
    # 기획서 §7.1 표: Train 약 3,764 · Val 941 · Calib 941 · Test 약 3,763
    assert make_split(generate(97), "pair", 0, MAIN).sizes() == {
        "train": 3764,
        "val": 941,
        "calib": 941,
        "test": 3763,
    }


def test_commutative_split_keeps_pairs_together():
    d = generate(97)
    s = make_split(d, "commutative", 0, MAIN)
    where = np.empty(d.n, dtype=object)
    for name, idx in s.indices.items():
        where[idx] = name
    swapped = d.b * d.p + d.a
    assert (where == where[swapped]).all()


def test_split_is_deterministic_and_seed_sensitive():
    d = generate(97)
    h0 = make_split(d, "pair", 0, MAIN).sha256()
    assert h0 == make_split(d, "pair", 0, MAIN).sha256()
    assert h0 != make_split(d, "pair", 1, MAIN).sha256()
    assert h0 != make_split(d, "commutative", 0, MAIN).sha256()


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        make_split(generate(7), "pair", 0, {"train": 0.5, "test": 0.4})


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_hash_reproduces(path):
    """설정에 적힌 해시가 다시 만든 분할과 같다 — 다른 PC · CI 에서도 (§7.1.1)."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    s = cfg["split"]
    split = make_split(generate(cfg["p"], cfg["op"]), s["method"], s["seed"], dict(s["fractions"]))
    assert split.sha256() == cfg["sha256"]


def test_operand_frequency_counts_both_operands():
    d = generate(5)
    idx = np.array([0 * 5 + 1, 1 * 5 + 1, 2 * 5 + 3])  # (0,1) (1,1) (2,3)
    f = operand_frequency(d, idx)
    # 0:1 1:3 2:1 3:1 4:0 → min 0, mean 1.2
    assert f["min"] == 0 and f["max"] == 3 and f["argmin"] == 4
    assert f["mean"] == pytest.approx(1.2) and f["flag"] is True


def test_main_train_split_frequency_not_flagged():
    d = generate(97)
    s = make_split(d, "pair", 0, MAIN)
    assert operand_frequency(d, s.indices["train"])["flag"] is False
