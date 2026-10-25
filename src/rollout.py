import numpy as np                         # 수치 계산
import pandas as pd                        # 결과 표 만들기

from environment import MAX_ASSET_WEIGHT, PortfolioEnv, action_to_weights   # 강화학습 환경 + 비중 변환


def build_eval_env(history_features, history_returns, eval_features, eval_returns, lookback=20):
    """평가 구간(val 또는 test)용 환경을 만든다.

    - 평가 첫날부터 '과거 N일 수익률'을 채우려고 직전 구간 끝의 (lookback-1)일을 앞에 붙인다.
      (붙이지 않으면 평가 구간 앞 19일을 그냥 버리게 됨. 직전 구간 값은 미래 정보가 아님)
    - 평가에서는 Safe-Guard(MDD 15% 조기종료)를 끈다: 전체 기간 성과를 같은 길이로 비교하기 위함.
      대신 summarize()가 MDD 15% 초과 여부를 따로 표시한다.
    """
    prefix = lookback - 1                                                    # 앞에 붙일 일수
    features = pd.concat([history_features.tail(prefix), eval_features])
    returns = pd.concat([history_returns.tail(prefix), eval_returns])
    return PortfolioEnv(features, returns, lookback=lookback, mdd_limit=10.0)


def build_val_env(data, lookback=20):
    """검증 구간 평가 환경: fit 구간 끝을 앞에 붙여 val 구간을 돈다. (후보 선택용)"""
    return build_eval_env(data.fit_features, data.fit_returns,
                          data.val_features, data.val_returns, lookback)


def build_test_env(data, lookback=20):
    """테스트 구간 평가 환경: 학습 구간 끝을 앞에 붙여 test 구간을 돈다. (최종 보고용)"""
    return build_eval_env(data.train_features, data.train_returns,
                          data.test_features, data.test_returns, lookback)


def predict_weights(model, obs, max_weight=MAX_ASSET_WEIGHT):
    """학습된 PPO가 관측 obs에서 정하는 목표 비중 (합=1, 각 자산 <= 0.4).

    환경 step()과 같은 action_to_weights()를 쓰므로 학습 때 비중과 백테스트 비중이 같다.
    evaluation-xai 백테스터에는 이 함수의 결과를 넘기면 된다.
    """
    action, _ = model.predict(obs, deterministic=True)
    return action_to_weights(action, max_weight)


def run_episode(env, action_fn, seed=42):
    """환경을 끝까지 한 번 돌리고 날짜별 기록을 표로 돌려준다.

    action_fn: 관측(obs)을 받아 행동(action)을 돌려주는 함수
               (학습된 PPO, 동일가중, 랜덤 등 무엇이든 가능)
    """
    obs, _ = env.reset(seed=seed)                                            # 처음 상태로
    records = []                                                              # 날짜별 기록
    while True:
        action = action_fn(obs)                                               # 행동 결정
        obs, reward, terminated, truncated, info = env.step(action)           # 하루 진행
        records.append({
            "date": info["date"],                                             # 날짜
            "net_return": info["net_return"],                                 # 비용 차감 후 일간 수익률
            "portfolio_value": info["portfolio_value"],                       # 자산가치
            "drawdown": info["drawdown"],                                     # 그날의 현재 낙폭
            "turnover": info["turnover"],                                     # 회전율
            "max_target_weight": float(info["target_weights"].max()),        # 그날 최대 자산 비중 (상한 확인용)
        })
        if terminated or truncated:                                           # 종료 조건
            break
    return pd.DataFrame(records).set_index("date")


def equal_weight_action_fn(n_assets):
    """동일가중 기준선: 행동을 항상 0으로 주면 softmax 결과가 1/N 비중이 된다."""
    zeros = np.zeros(n_assets, dtype=np.float32)
    return lambda obs: zeros


def summarize(df, mdd_limit=0.15):
    """날짜별 기록에서 핵심 성과를 계산한다. (12개 지표 전체는 추후 evaluate.py에서)"""
    r = df["net_return"].to_numpy()                                           # 일간 수익률
    std = r.std(ddof=1) if len(r) > 1 else np.nan                             # 일간 변동성
    max_mdd = float(df["drawdown"].max())                                     # 최대낙폭 = 현재 낙폭의 최댓값
    return {
        "days": len(df),                                                      # 평가 일수
        "cum_return": float(np.prod(1.0 + r) - 1.0),                          # 누적 수익률 (일간 수익률을 복리로 곱함)
        "ann_vol": float(std * np.sqrt(252)),                                 # 연환산 변동성
        "sharpe": float(r.mean() / std * np.sqrt(252)) if std and std > 0 else np.nan,  # 연환산 샤프(무위험 0)
        "max_mdd": max_mdd,                                                   # 최대낙폭
        "breach_15pct": bool(max_mdd > mdd_limit),                            # Safe-Guard 기준 초과 여부
        "avg_turnover": float(df["turnover"].mean()),                         # 평균 회전율
        "max_weight": float(df["max_target_weight"].max()),                   # 기간 중 최대 자산 비중 (<= 0.4 이어야 함)
    }
