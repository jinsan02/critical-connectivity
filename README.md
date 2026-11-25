# critical-connectivity

**Is there an architecture-invariant critical connectivity?** — 신경망과 소형 모델 군집의 창발적 일반화(Grokking)에 아키텍처와 상관없는 임계 연결 밀도가 있는지 검정하는 연구 저장소입니다.

- 같은 저장 파라미터(64M)에서 **Dense-64M**, 8M 모델 8개를 Top-1 Router 로 잇는 **Cross-Model**, **Parallel-Mixing** 을 비교합니다.
- 주 과제는 모듈러 덧셈 $a+b \bmod 97$ (9,409쌍, 입력 `a b =`)입니다.
- 주 가설: 교차 연결 밀도 ρ 에 따른 Grokking 확률 · 시점 곡선에 변화점이 있다(H3), 유효 연결량 $C_{\mathrm{eff}}$ 가 기존 예측식을 넘는 설명력을 가진다(H2).

설계 전체는 [`docs/plan/research_plan_v0.4.md`](docs/plan/research_plan_v0.4.md) 에 있습니다.

## 현재 상태

**Gate 1 전.** 저장소 규칙 · 훅 · CI 를 세웠고 연구 코드는 아직 없습니다. 진행 상황은 [`STATE.md`](STATE.md).

## 처음 받았을 때

```powershell
git clone https://github.com/jinsan02/critical-connectivity.git
cd critical-connectivity
git config core.hooksPath .githooks
powershell -NoProfile -ExecutionPolicy Bypass -File env\create-critconn.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File verify.ps1
```

## 문서

| 문서 | 내용 |
| --- | --- |
| [`STATE.md`](STATE.md) | 진행 상태 · 결정 기록 · 열린 질문 |
| [`STRUCTURE.md`](STRUCTURE.md) | 폴더 지도 · 단계별 기준 문서 · 이름 규칙 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 이슈 · 브랜치 · 커밋 · PR |
| [`AGENTS.md`](AGENTS.md) | AI 에이전트 작업 지침 |
| [`docs/`](docs/README.md) | 규칙 · 환경 · 기획서 |
