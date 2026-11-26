"""데이터 생성 · 분할 (기획서 v0.5 §7)."""

from src.datasets.modular_addition import ModularData, generate
from src.datasets.split import Split, SplitSpec, make_split, operand_frequency

__all__ = ["ModularData", "Split", "SplitSpec", "generate", "make_split", "operand_frequency"]
