# RL Engine

에이전틱 RAG + 강화학습 기반 로보어드바이저 프로젝트의 RL(강화학습) 파트다. PPO 기반 동적 자산배분 엔진을 구현하고, 보상 함수 설계에 따른 성과 차이를 실험·검증한다. 프로젝트 전체 요건은 CLAUDE.md 참고.

## 폴더 구조

```
rl-engine/
├── src/
│   ├── environment.py       # Gymnasium 커스텀 거래 환경 (PortfolioEnv)
│   ├── data_loader.py        # walk-forward 윈도우 로드 + fit/val/test 분리
│   ├── rollout.py            # 평가 환경, predict_weights, 성과 요약
│   ├── make_dummy_data.py    # data-pipeline과 같은 형식의 가짜 데이터 생성
│   ├── train.py              # 단일 보상함수 PPO 학습 (train_one)
│   ├── train_compare.py      # 보상함수 3종(5개 변형) 비교 실험 (train_one 반복 호출)
│   ├── test_environment.py   # 환경 규격 검증 + 랜덤 에이전트 동작 테스트
│   ├── metrics.py            # 12개 성과지표 계산 (numpy/pandas만 사용)
│   ├── test_metrics.py       # 성과지표·weights_to_action 검증
│   └── evaluate.py           # Walk-Forward 백테스트 (저장된 PPO 모델 + 비교군, 12개 지표)
├── data/                      # data-pipeline 리포의 features.parquet / returns.parquet 수신 위치 (git 제외)
├── models/                    # 학습된 PPO 모델 저장 (.zip은 git 제외, 요약 CSV만 추적)
├── logs/                      # tensorboard 학습 로그 (git 제외)
├── requirements.txt
└── README.md
```

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

# 4. 단일 보상함수 PPO 학습 (기본: fit 학습 -> val 평가)
python train.py --reward mdd --mdd-lambda 1.0

# 5. 보상함수 3종(5개 변형) 비교 실험
python train_compare.py --stage val  --windows 1 2 3 4 --seeds 42 43 44   # 선택용 (val)
python train_compare.py --stage test --windows 1 2 3 4 --seeds 42 43 44   # 최종 보고용 (test)

# 6. Walk-Forward 백테스트 + 12개 성과지표 (5번 test 단계 모델을 불러와 평가만 함)
python evaluate.py --stage test --windows 1 2 3 4 --seeds 42 43 44
python evaluate.py --baselines-only        # 모델 없이 동일가중 기준선만
python test_metrics.py                     # 지표 계산 검증

# 7. 학습 곡선 확인
tensorboard --logdir ../logs
```

## 강화학습 환경 설계

### 관측 공간 (Observation Space)
`data-pipeline`에서 공급하는 종목별 피처(`{ticker}_return`, `{ticker}_rsi`, `{ticker}_macd`)에 현재 포트폴리오 비중을 더해 구성한다. 현재 비중을 포함하는 이유는 에이전트가 거래비용을 인식하고 학습하도록 하기 위함이다.

### 행동 공간 (Action Space)
연속형(Continuous)으로 설계했다. PPO가 출력한 임의의 실수 벡터를 softmax로 변환해 합이 1이 되는 자산별 비중으로 만든 뒤, **자산별 상한 0.4**로 투영한다(넘친 비중은 상한 아래 자산들에 기존 비중 비례로 재분배, 반복). 상한값은 팀 설계 문서(`backtest.md`, `decisions.md`)와 evaluation-xai `config.max_asset_weight`와 같다.

환경(`step`)과 평가(`rollout.predict_weights`)는 같은 함수 `environment.action_to_weights()`를 쓰므로 학습 때 비중과 백테스트 비중이 어긋나지 않는다.

### 거래비용
수수료 0.015%, 슬리피지 0.05%를 명세 기준값으로 반영했다. 비중 변화량(회전율)에 비례해 비용을 차감한다.

### Safe-Guard
현재 낙폭이 15%를 초과하면 에피소드를 조기 종료한다.

MDD 페널티 보상(`r_t - λ·DD_t`)은 거의 항상 음수라서, 종료 후 보상이 0이면 "일부러 15%를 깨고 빨리 끝내기"가 누적 보상이 더 높아진다. 이를 막기 위해 Safe-Guard 종료 시 `λ × 0.15 × (남은 스텝 수)`를 한 번에 차감한다(= 남은 기간 내내 낙폭이 한도에 머문 것으로 취급). 끝까지 버티는 정책은 매 스텝 `DD_t ≤ 0.15`이므로 조기 종료로 페널티를 줄일 수 없다. 명세의 보상 수식 자체는 바꾸지 않았다. 학습 시 `VecNormalize`의 보상 클리핑은 이 페널티가 잘리지 않도록 꺼 두었다(`clip_reward=np.inf`).

## 학습 / 검증 / 테스트 분리

각 walk-forward 윈도우의 학습 구간 마지막 6개월을 검증(val) 구간으로 뗀다(`validation.md` 게이트 3).

```
[train_start ..... val_start ..... train_end)[train_end ..... test_end)
|<----- fit ----->|<---- val ---->|          |<------ test ------>|
```

- `--stage val`: fit 구간에서 학습 → val 구간에서 평가. 보상형·λ·하이퍼파라미터 선택은 여기서만 한다.
- `--stage test`: 고른 설정으로 train 전체(fit+val)를 재학습 → test 구간에서 평가. 리포트·ANOVA용 최종 성과.

## 보상 함수 3종 실험

| 변형 | 수식 | 비고 |
|---|---|---|
| 단순 수익률 | `reward = r_t` | 위험을 고려하지 않는 베이스라인 |
| 샤프비율 기반 | `reward = r_t / (sigma_t + eps)` | 최근 20일 변동성으로 나눠 위험조정 |
| MDD 페널티 결합 | `reward = r_t - lambda * MDD_t` | lambda ∈ {0.5, 1.0, 5.0}으로 공격적~방어적 성향 비교. `MDD_t`는 시점 t의 현재 낙폭(코드 변수명 `current_drawdown`) |

다섯 가지 변형 모두 동일 시드(seed=42)와 동일 학습 스텝(100,000)으로 학습해 공정하게 비교한다.

## 평가 (evaluate.py)

`train_compare.py`가 저장한 모델(`models/ppo_{stage}_w{n}_{변형}_s{seed}.zip`)을 불러와 각 윈도우의 평가 구간을 결정론적으로 돌린다. 학습과 같은 환경(`PortfolioEnv`, 같은 거래비용·비중 상한)을 쓰고 Safe-Guard만 끈다.

- 윈도우 1~4의 test 구간(2022~2025)은 겹치지 않으므로 이어붙여 **전체 out-of-sample 기간** 성과도 계산한다.
- 정보비율의 벤치마크는 동일가중(EqualWeight).
- 일부 윈도우 모델이 빠진 (전략, 시드)는 기간이 달라 시드 평균에서 제외하고 경고를 띄운다.

| # | 지표 | 키 | # | 지표 | 키 |
|---|---|---|---|---|---|
| 1 | 누적 수익률 | `cum_return` | 7 | 칼마비율 | `calmar` |
| 2 | 연환산 수익률 | `cagr` | 8 | 승률(일간) | `win_rate` |
| 3 | 연환산 변동성 | `ann_vol` | 9 | VaR 95% (일간) | `var_95` |
| 4 | 샤프비율 | `sharpe` | 10 | CVaR 95% (일간) | `cvar_95` |
| 5 | 소르티노비율 | `sortino` | 11 | 평균 회전율 | `avg_turnover` |
| 6 | 최대낙폭(MDD) | `max_mdd` | 12 | 정보비율(vs 동일가중) | `information_ratio` |

결과 CSV (`models/`): `eval_{stage}_windows.csv`(윈도우별), `eval_{stage}_walkforward.csv`(전체 기간), `eval_{stage}_summary.csv`(시드 평균±표준편차), `eval_{stage}_daily.csv`(날짜별 수익률, 그래프·ANOVA용).

비교군은 `evaluate.BASELINES`에 등록한다. MVO처럼 비중을 직접 정하는 전략은 `env.simple_returns[: env.current_step + 1]`(오늘까지의 실제 수익률)로 비중을 구한 뒤 `environment.weights_to_action(비중)`으로 행동을 돌려주면 같은 거래비용·상한으로 평가된다.

## 데이터 연동 (진행 중)

`data-pipeline` 리포의 `features.parquet` / `returns.parquet`를 받아 연동 예정이다. 현재까지는 랜덤 생성한 가짜 데이터로 환경 동작과 PPO 학습 파이프라인을 검증한 상태이며, 실제 데이터 연동 시 다음을 반영해야 한다:

- `prices` 기반이 아닌 이미 계산된 `returns` 기반으로 환경 입력 구조 변경
- `{ticker}_*` 컬럼명 패턴에서 12개 종목을 자동 파싱하는 로직 추가
- 정규화는 `data-pipeline`에서 `TRAIN_END_DATE` 이전 구간 통계로만 fit했음을 확인(Data Leakage 방지 조건 충족)

## 알려진 한계점

- 현재까지 PPO 학습·보상함수 비교는 전부 랜덤 가짜 데이터 기준이며, 실제 시장 데이터의 통계적 특성(두꺼운 꼬리, 변동성 군집 등)은 반영되지 않았다.
- MVO 비교군은 아직 구현 전이다 (`evaluate.BASELINES`에 등록 자리만 있음).
- 12개 지표는 일반적인 구성으로 정했다. 팀 명세(`backtest.md`)의 지표 목록과 다르면 `metrics.py`에서 맞춘다.
- SHAP 기반 의사결정 해석, RAG Risk Tag 연동 인터페이스는 추후 단계에서 작업 예정이다.

> 본 프로젝트는 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없다. 백테스팅 성과가 미래 수익을 보장하지 않는다.
