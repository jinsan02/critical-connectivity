# STRUCTURE.md — 무엇이 어디 있나

## 1. 읽는 순서

- **사람:** `README.md` → `STATE.md` → `docs/plan/research_plan_v0.4.md` §1~§3 → `CONTRIBUTING.md`
- **AI 에이전트:** `AGENTS.md` 2절 순서 (`STATE` → `STRUCTURE` → `docs/RULES` → 폴더 `README`·`AGENTS` → 기준 문서 해당 절)

## 2. 폴더 트리

혼자 운영하므로 담당은 모두 `@jinsan02` 다 (`.github/CODEOWNERS`). **(예정)** 은 아직 비어 있다. **[git 밖]** 은 `.gitignore` 와 pre-commit 이 막는다.

```
critical-connectivity/
├─ AGENTS.md · CLAUDE.md       AI 작업 지침 (CLAUDE.md = @AGENTS.md)
├─ README.md · STATE.md · STRUCTURE.md · CONTRIBUTING.md
├─ verify.ps1                  커밋 전 전체 검증
├─ pyproject.toml              ruff · pytest 설정
├─ .githooks/                  commit-msg · pre-commit (본체는 tools/)
├─ .github/                    CI · PR · 이슈 템플릿 · CODEOWNERS
├─ docs/
│  ├─ RULES.md                 하드룰 — 규칙의 단일 출처
│  ├─ COMMIT_CONVENTION.md · WORKFLOW.md · ENVIRONMENT.md · LEDGER_SCHEMA.md
│  ├─ plan/                    기준 기획서 v0.4 + 조사 문서
│  └─ 지난설계/                 v0.1~v0.3 (구현 기준 아님)
├─ env/                        conda 선언 · 설치 로그 · 하드웨어 실측
├─ configs/                    (예정) 실험 설정 YAML
├─ src/                        (예정) datasets · models · connectivity · training · analysis
├─ scripts/                    (예정) 실행 진입점
├─ experiments/                LEDGER.tsv · runs/<run_id>/
├─ reports/                    보고서 · 그림
├─ tools/ · tests/             훅 본체 · 검사기 · 테스트
├─ data/                       [git 밖] 생성 데이터
├─ checkpoints/                [git 밖] 체크포인트 (기획서 §15.2, 약 370GB)
├─ results/metrics/ · activations/ · graphs/   [git 밖] 지표 Parquet 등
└─ lockbox/                    [git 밖] 확증 단계 전용 (RULES 3)
```

## 3. 단계별 기준 문서

| 단계 | 기준 (이 순서로 이김) | 만드는 곳 | 커밋 scope |
| --- | --- | --- | --- |
| 환경 | `docs/ENVIRONMENT.md` → 기획서 §14 | `env/` | `env` |
| 데이터 | 기획서 §7, §7.1.1 → `docs/plan/dataset_search_modular.md` | `src/datasets/` | `data` |
| 모델 | 기획서 §5, §12A-2 | `src/models/` | `model` |
| 연결 · Router | 기획서 §5.4~§5.7, §6 | `src/connectivity/` | `conn` |
| 학습 | 기획서 §14, §15 | `src/training/` | `train` |
| 분석 | 기획서 §10, §12A, §12B | `src/analysis/` | `analysis` |
| 실험 실행 | 기획서 §9, §15.1 → `docs/WORKFLOW.md` | `configs/`, `experiments/` | `exp` |
| 확장 연구 | 기획서 §23~§27 | (예정) | `ext` |
| 훅 · CI | `docs/RULES.md` → `docs/COMMIT_CONVENTION.md` | `tools/`, `.github/` | `infra` |

## 4. 큰 실수를 막는 체크리스트

| 확인 | 어기면 생기는 일 |
| --- | --- |
| 비교 묶음에서 한 변수만 다른가 | 차이의 원인을 가를 수 없다 |
| 조율에 Validation 만 썼는가 | Test 가 오염되어 일반화 주장이 무너진다 |
| H2 지표를 $t_0$ 에서만 계산했는가 | 결과로 원인을 설명하는 순환논리 |
| Cross-Model VRAM 을 64M 기준으로 잡았는가 | 동시 실행 중 OOM |
| Logits · Softmax 가 FP32 인가 | Softmax Collapse 로 Grokking 이 막힘 [25] |
| 실패 run 도 원장에 남겼는가 | 같은 실패 반복, 생존 분석의 중도절단 누락 |
| Gate 2.5 이후 변경을 `deviation_log` 에 썼는가 | 사전등록 무효 |

## 5. 이름 규칙

| 대상 | 규칙 | 예 |
| --- | --- | --- |
| 파일 · 폴더 · 식별자 | 영어 소문자, `_` | `cross_model.py` |
| run_id | (권장, Gate 1 에서 확정) `<phase>_<exp>_<arch>_<변형>_i<init>m<mask>o<order>` | `p2_exp02_cross_r0.04_i1m2o0` |
| 브랜치 | `exp/<phase>-<slug>` · `feat/<scope>-<slug>` · `fix/<slug>` · `docs/<slug>` | `exp/p0-r1-anchor` |
| 커밋 | `<type>(<scope>): <제목>` | `feat(data): 모듈러 덧셈 생성기와 분할을 더한다` |
| 문서 | 기준 문서는 대문자(`RULES.md`), 기록은 `<주제>-YYYYMMDD.md` | `docs/throughput-20261126.md` |
