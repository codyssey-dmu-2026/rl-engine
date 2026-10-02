# RL Engine

에이전틱 RAG + 강화학습 기반 로보어드바이저 프로젝트의 RL(강화학습) 파트다. PPO 기반 동적 자산배분 엔진을 구현하고, 보상 함수 설계에 따른 성과 차이를 실험·검증한다. 프로젝트 전체 요건은 CLAUDE.md 참고.

## 폴더 구조

rl-engine/
├── src/
│ ├── environment.py # Gymnasium 커스텀 거래 환경 (PortfolioEnv)
│ ├── train.py # 단일 보상함수 PPO 학습 스크립트
│ ├── train_compare.py # 보상함수 3종(5개 변형) 비교 실험
│ ├── test_environment.py # 환경 규격 검증 + 랜덤 에이전트 동작 테스트
│ └── evaluate.py # Walk-Forward 백테스트 + 12개 성과지표 (예정)
├── data/ # data-pipeline 리포의 features.parquet / returns.parquet 수신 위치 (git 제외)
├── models/ # 학습된 PPO 모델 저장 (.zip은 git 제외, 요약 CSV만 추적)
├── logs/ # tensorboard 학습 로그 (git 제외)
├── requirements.txt
└── README.md


## 실행 방법

```bash
# 1. 가상환경 준비 (Python 3.10 이상)
python -m venv venv
source venv/bin/activate      # 윈도우: venv\Scripts\activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 규격 검증 (가짜 데이터 기준)
cd src
python test_environment.py

# 4. 단일 보상함수 PPO 학습
python train.py

# 5. 보상함수 3종(5개 변형) 비교 실험
python train_compare.py

# 6. 학습 곡선 확인
tensorboard --logdir ../logs
```

## 강화학습 환경 설계

### 관측 공간 (Observation Space)
`data-pipeline`에서 공급하는 종목별 피처(`{ticker}_return`, `{ticker}_rsi`, `{ticker}_macd`)에 현재 포트폴리오 비중을 더해 구성한다. 현재 비중을 포함하는 이유는 에이전트가 거래비용을 인식하고 학습하도록 하기 위함이다.

### 행동 공간 (Action Space)
연속형(Continuous)으로 설계했다. PPO가 출력한 임의의 실수 벡터를 softmax로 변환해 합이 1이 되는 자산별 비중으로 만든다.

### 거래비용
수수료 0.015%, 슬리피지 0.05%를 명세 기준값으로 반영했다. 비중 변화량(회전율)에 비례해 비용을 차감한다.

### Safe-Guard
MDD(최대낙폭)가 15%를 초과하면 에피소드를 조기 종료한다.

## 보상 함수 3종 실험

| 변형 | 수식 | 비고 |
|---|---|---|
| 단순 수익률 | `reward = r_t` | 위험을 고려하지 않는 베이스라인 |
| 샤프비율 기반 | `reward = r_t / (sigma_t + eps)` | 최근 20일 변동성으로 나눠 위험조정 |
| MDD 페널티 결합 | `reward = r_t - lambda * MDD_t` | lambda ∈ {0.5, 1.0, 5.0}으로 공격적~방어적 성향 비교 |

다섯 가지 변형 모두 동일 시드(seed=42)와 동일 학습 스텝(100,000)으로 학습해 공정하게 비교한다.

## 데이터 연동 (진행 중)

`data-pipeline` 리포의 `features.parquet` / `returns.parquet`를 받아 연동 예정이다. 현재까지는 랜덤 생성한 가짜 데이터로 환경 동작과 PPO 학습 파이프라인을 검증한 상태이며, 실제 데이터 연동 시 다음을 반영해야 한다:

- `prices` 기반이 아닌 이미 계산된 `returns` 기반으로 환경 입력 구조 변경
- `{ticker}_*` 컬럼명 패턴에서 12개 종목을 자동 파싱하는 로직 추가
- 정규화는 `data-pipeline`에서 `TRAIN_END_DATE` 이전 구간 통계로만 fit했음을 확인(Data Leakage 방지 조건 충족)

## 알려진 한계점

- 현재까지 PPO 학습·보상함수 비교는 전부 랜덤 가짜 데이터 기준이며, 실제 시장 데이터의 통계적 특성(두꺼운 꼬리, 변동성 군집 등)은 반영되지 않았다.
- `evaluate.py`(Walk-Forward 백테스트, 12개 성과지표, MVO·동일가중 비교군)는 아직 구현 전이다.
- SHAP 기반 의사결정 해석, RAG Risk Tag 연동 인터페이스는 추후 단계에서 작업 예정이다.

> 본 프로젝트는 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없다. 백테스팅 성과가 미래 수익을 보장하지 않는다.
