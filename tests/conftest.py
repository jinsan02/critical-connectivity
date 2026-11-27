"""pytest 공통 설정."""

import os

# 본체 critconn 의 MKL(INTEL 스레딩)이 llvm-openmp 의 libiomp5md 에서 심볼을 못 찾아
# numpy 행렬곱이 0xC06D007F 로 죽는다. numpy 를 처음 불러오기 전에 정해야 한다 (T-012).
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
