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
│   ├── model_meta.py         # 모델 옆 메타 정보(.json) 저장·호환성 검사
│   ├── metrics.py            # 명세 12개 성과지표 + 보조지표 (numpy/pandas만 사용)
│   ├── evaluate.py           # Walk-Forward 백테스트 (저장된 PPO 모델 + 비교군, 12개 지표)
│   ├── test_environment.py   # [pytest] 데이터 로더·환경·비중 변환·Safe-Guard·메타 검증
│   └── test_metrics.py       # [pytest] 성과지표·weights_to_action 검증
├── data/                      # data-pipeline 결과물 (git 제외)
│   ├── processed/             #   실데이터 (기본값). 없으면 실행이 멈춘다
│   └── dummy/                 #   make_dummy_data.py 가짜 데이터 (--data-dir 로 직접 지정할 때만)
├── models/                    # 학습된 PPO 모델 (.zip은 git 제외, 결과 CSV와 메타 .json은 추적)
├── logs/                      # tensorboard 학습 로그 (git 제외)
├── pytest.ini                 # 루트에서 `pytest` 한 번으로 src/ 테스트 실행
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

# 3. 테스트 (프로젝트 루트에서). 가짜 데이터는 임시 폴더에 자동 생성,
#    data/processed 실데이터가 있으면 실데이터 테스트도 함께 실행
pytest

cd src
# (데이터 폴더 규칙) 기본은 data/processed 실데이터. 없으면 에러로 멈춘다.
# 가짜 데이터로 동작만 확인하려면: python make_dummy_data.py 후 --data-dir ../data/dummy 를 붙인다.

# 4. 단일 보상함수 PPO 학습 (기본: fit 학습 -> val 평가)
python train.py --reward mdd --mdd-lambda 1.0

# 5. 보상함수 3종(5개 변형) 비교 실험
python train_compare.py --stage val  --windows 1 2 3 4 --seeds 42 43 44   # 선택용 (val)
python train_compare.py --stage test --windows 1 2 3 4 --seeds 42 43 44   # 최종 보고용 (test)

# 6. Walk-Forward 백테스트 + 12개 성과지표 (5번 test 단계 모델을 불러와 평가만 함)
python evaluate.py --stage test --windows 1 2 3 4 --seeds 42 43 44
python evaluate.py --baselines-only        # 모델 없이 동일가중 기준선만
python evaluate.py --benchmark 069500      # 알파·베타·IR 기준을 바꿀 때 (기본 SPY)

# 7. 학습 곡선 확인
tensorboard --logdir ../logs
```

## 강화학습 환경 설계

### 관측 공간 (Observation Space)
`data-pipeline`에서 공급하는 종목별 피처에 현재 포트폴리오 비중을 더해 구성한다. 자산 12개 기준 **288차원**이다.

| 구성 | 차원 |
|---|---|
| 과거 20일 정규화 수익률 `{ticker}_return` | 20 × 12 = 240 |
| 오늘의 `{ticker}_rsi`, `{ticker}_macd`, `{ticker}_is_market_open` | 36 |
| 현재 포트폴리오 비중 | 12 |
| (선택) RiskTag `{ticker}_risk` | +12 → 300 |

현재 비중을 포함하는 이유는 에이전트가 거래비용을 인식하고 학습하도록 하기 위함이다. 자산 순서는 `returns.parquet` 컬럼 순서(문자열 정렬)이며, 가짜 데이터도 같은 순서로 만든다.

#### RiskTag 관측 (선택)
`--risk-file risk.parquet`을 주면 `{ticker}_risk` 12개 컬럼을 오늘 피처 뒤에 붙인다(288 → 300차원). 파일 형식은 index=날짜, 컬럼=`{ticker}_risk`이고, 날짜가 없는 행은 0으로 채우며 개수를 경고한다. **각 행은 그 날짜의 결정 시각에 이미 알 수 있던 태그만으로 만들어야 한다.** 결정 시각 규칙(KST 장마감 / NY 16:00 등)은 agentic-research와 합의한 것을 따르며, 파일 생성 쪽 책임이다. RiskTag를 넣고 학습한 모델은 평가 때도 같은 파일을 줘야 한다(메타 검사로 강제).

### 행동 공간 (Action Space)
연속형(Continuous)으로 설계했다. PPO가 출력한 임의의 실수 벡터를 softmax로 변환해 합이 1이 되는 자산별 비중으로 만든 뒤, **자산별 상한 0.4**로 투영한다(넘친 비중은 상한 아래 자산들에 기존 비중 비례로 재분배, 반복). 상한값은 팀 설계 문서(`backtest.md`, `decisions.md`)와 evaluation-xai `config.max_asset_weight`와 같다.

환경(`step`)과 평가(`rollout.predict_weights`)는 같은 함수 `environment.action_to_weights()`를 쓰므로 학습 때 비중과 백테스트 비중이 어긋나지 않는다.

### 거래비용
수수료 0.015%, 슬리피지 0.05%를 명세 기준값으로 반영했다. 비중 변화량(회전율)에 비례해 비용을 차감한다.

### Safe-Guard
현재 낙폭이 15%를 초과하면 에피소드를 조기 종료한다.

MDD 페널티 보상(`r_t - λ·DD_t`)은 거의 항상 음수라서, 종료 후 보상이 0이면 "일부러 15%를 깨고 빨리 끝내기"가 누적 보상이 더 높아진다. 이를 막기 위해 Safe-Guard 종료 시 "남은 기간 내내 낙폭이 한도(15%)에 머문 것"의 벌점을 PPO 할인율(γ=0.99)대로 합해 한 번에 차감한다.

```
penalty = λ × 0.15 × (1 + γ + γ² + ... + γ^(남은 스텝-1))   ≤ λ × 0.15 / (1-γ) = λ × 15
```

끝까지 버티는 정책은 매 스텝 `DD_t ≤ 0.15`이므로, PPO가 최대화하는 할인 누적 보상 기준으로 조기 종료가 이득이 될 수 없다. 할인을 반영해 페널티 크기가 λ=5에서도 75 이하로 묶여 보상 정규화가 안정적이다(이전 방식 `λ × 0.15 × 남은 스텝`은 λ=5에서 수백). 테스트는 가짜 데이터와 실데이터 4개 윈도우 × λ∈{0.5, 1, 5}에서 "미래를 보고 일부러 15%를 깨는 정책"이 동일가중보다 할인 누적 보상이 낮음을 확인한다. 명세의 보상 수식 자체는 바꾸지 않았다. 학습 시 `VecNormalize`의 보상 클리핑은 이 페널티가 잘리지 않도록 꺼 두었다(`clip_reward=np.inf`).

평가(`evaluate.py`)에서는 Safe-Guard를 끄고 평가 구간 전체를 돈다. 대신 `breach_15pct`로 초과 여부를 표시한다. evaluation-xai 백테스터는 15% 초과 시 청산하므로, 리포트 수치를 어느 쪽 기준으로 낼지는 팀 결정 사항이다.

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

**`MDD_t`를 "현재 낙폭"으로 해석한 근거**: 기간 최대낙폭(running max)을 쓰면 한 번 크게 빠진 뒤에는 무엇을 해도 벌점이 줄지 않아, 회복하려는 행동에 신호가 사라진다. 현재 낙폭은 고점을 회복하면 0으로 돌아오므로 "낙폭을 줄이는 행동"에 매일 보상 신호가 생긴다. 시점 t까지 보면 현재 낙폭의 최댓값이 곧 MDD이므로 같은 위험 개념을 스텝 단위로 쪼갠 것이다.

다섯 가지 변형 모두 같은 시드 목록과 같은 학습 스텝(100,000)으로 학습해 공정하게 비교한다. ANOVA 검증 1(보상 함수별 비교)에는 변형마다 여러 표본이 필요하므로 `--seeds 42 43 44`처럼 시드를 여러 개 돌린다.

학습한 모델마다 같은 이름의 `.json` 메타 정보(자산 순서, 관측 차원, lookback, RiskTag 컬럼, 데이터 폴더, 가짜 데이터 여부, 학습 구간)를 남긴다. `evaluate.py`는 이 정보로 모델과 평가 데이터가 맞는지 검사하고, 다르면 멈춘다(가짜 데이터로 학습한 모델은 `--allow-dummy` 없이는 거부).

## 평가 (evaluate.py)

`train_compare.py`가 저장한 모델(`models/ppo_{stage}_w{n}_{변형}_s{seed}.zip`)을 불러와 각 윈도우의 평가 구간을 결정론적으로 돌린다. 학습과 같은 환경(`PortfolioEnv`, 같은 거래비용·비중 상한)을 쓰고 Safe-Guard만 끈다.

- 윈도우 1~4의 test 구간(2022~2025)은 겹치지 않으므로 이어붙여 **전체 out-of-sample 기간** 성과도 계산한다.
- 알파·베타·정보비율의 벤치마크는 `--benchmark`로 정한다. 기본은 **SPY**(S&P500 추종 ETF)다. 데이터에 지수 원본이 없어 ETF로 대용하며, `069500`(KODEX 200, KOSPI200 추종)이나 `EqualWeight`도 쓸 수 있다.
- 일부 윈도우 모델이 빠진 (전략, 시드)는 기간이 달라 시드 평균에서 제외하고 경고를 띄운다.

명세의 12개 지표 (`metrics.METRIC_NAMES`):

| 구분 | 지표 | 키 |
|---|---|---|
| 수익률 | 누적 수익률, CAGR | `cum_return`, `cagr` |
| 리스크 | 연환산 변동성, VaR(95%, 일간), CVaR(95%, 일간), MDD | `ann_vol`, `var_95`, `cvar_95`, `max_mdd` |
| 위험조정 | 샤프, 소르티노, 칼마 | `sharpe`, `sortino`, `calmar` |
| 상대성과 (벤치마크 대비) | 알파(OLS 절편, 연환산), 베타(OLS 기울기), 정보비율 | `alpha`, `beta`, `information_ratio` |

보조 지표로 `win_rate`(일간 승률), `avg_turnover`(평균 회전율)도 함께 저장한다. 무위험수익률은 0이다(팀 결정 시 `compute_metrics(rf=...)`로 변경). 정의는 evaluation-xai `metrics.py`와 같게 맞췄으므로, 최종 리포트 수치는 한쪽으로 통일해서 쓴다.

결과 CSV (`models/`): `eval_{stage}_windows.csv`(윈도우별), `eval_{stage}_walkforward.csv`(전체 기간), `eval_{stage}_summary.csv`(시드 평균±표준편차), `eval_{stage}_daily.csv`(날짜별 수익률, 그래프·ANOVA용).

비교군은 `evaluate.BASELINES`에 등록한다. MVO는 evaluation-xai(`baselines.py`)에 명세대로 구현돼 있으므로 여기서 따로 만들지 않는다(두 구현의 숫자가 달라지는 것을 막기 위함). rl-engine 환경에서 같이 돌려야 하면, 그 함수가 낸 비중을 `environment.weights_to_action(비중)`으로 행동으로 바꿔 등록하면 같은 거래비용·상한으로 평가된다.

### 다른 레포에서 쓰는 법 (SHAP·API)
- `environment.action_to_weights(action)`: `(N,)`과 `(batch, N)` 입력을 모두 받는다. SHAP처럼 여러 관측을 한 번에 넣어도 행마다 같은 결과가 나온다.
- `rollout.predict_weights(model, obs)`: PPO 정책의 목표 비중(합=1, 각 ≤ 0.4). obs도 `(batch, 288)` 가능.
- `model_meta.load_meta(모델경로)`, `check_compatible(...)`: 모델을 불러오기 전 자산 순서·관측 차원 확인.

## 데이터 연동

`data-pipeline`의 `data/processed/`(`features_window{1..4}.parquet`, `returns.parquet`, `walk_forward_manifest.json`)를 그대로 읽는다. `data/`는 git에 올리지 않으므로 팀 공용 데이터 스냅샷을 받아 `data/processed/`에 넣는다.

- 환경 입력은 이미 계산된 로그수익률(`returns.parquet`, 정규화 전)과 정규화 피처(`features_window{n}`)다.
- 종목은 `{ticker}_return` 컬럼에서 자동으로 읽는다(12개, 문자열 정렬 순서).
- 정규화는 `data-pipeline`에서 학습 구간 통계로만 fit했다(Data Leakage 방지).

## 알려진 한계점 / 팀 결정 대기

- Walk-Forward 창은 data-pipeline manifest를 따른다. 현재는 학습 시작이 2018-01-01로 고정된 확장형(4/5/6/7년)이며, 명세의 "학습 4년 → 테스트 1년 롤링"과 다르다(data-pipeline에서 결정).
- 벤치마크는 지수 원본이 없어 SPY(또는 069500) ETF로 대용한다. 무위험수익률은 0.
- 평가 시 Safe-Guard 처리(rl-engine은 끄고 표시, evaluation-xai는 15% 초과 시 청산)는 팀이 하나로 정해야 한다.
- RiskTag 관측 입력 자리는 있지만, 태그 → `{ticker}_risk` 파일을 만드는 규칙(결정 시각, "자료 없음"과 "위험 없음" 구분)은 agentic-research와 합의 후 확정한다. Safe-Guard 비중 축소 규칙(θ)도 같다.
- 학습된 모델(.zip)은 git에 없으므로 공유 경로(드라이브·릴리스)를 팀에서 정해야 한다.

> 본 프로젝트는 교육 목적으로 개발되며 실제 투자 조언에 사용할 수 없다. 백테스팅 성과가 미래 수익을 보장하지 않는다.
