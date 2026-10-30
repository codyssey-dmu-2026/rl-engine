"""성과지표 계산 (numpy/pandas만 사용 -> PPO/torch 없이도 테스트 가능).

모든 지표는 '비용 차감 후 일간 단순수익률' 시계열 하나(+ 벤치마크 수익률)로 계산한다.
그래서 PPO, 동일가중, MVO 등 어떤 전략이든 같은 함수로 공정하게 비교할 수 있다.

[명세 12개 지표] (과제 명세 5. 백테스트 및 성과 평가)
| 구분     | 키                 | 의미                                         | 좋은 방향 |
|----------|--------------------|----------------------------------------------|-----------|
| 수익률   | cum_return         | 누적 수익률 (복리)                            | 높을수록  |
|          | cagr               | 연환산 수익률 (CAGR)                          | 높을수록  |
| 리스크   | ann_vol            | 연환산 변동성                                 | 낮을수록  |
|          | var_95             | 일간 VaR 95% (역사적, 손실을 양수로)           | 낮을수록  |
|          | cvar_95            | 일간 CVaR 95% (VaR 넘는 손실의 평균)           | 낮을수록  |
|          | max_mdd            | 최대낙폭 (MDD)                                | 낮을수록  |
| 위험조정 | sharpe             | 연환산 샤프비율 (무위험수익률 rf 차감)         | 높을수록  |
|          | sortino            | 연환산 소르티노비율 (하방 변동성 기준)         | 높을수록  |
|          | calmar             | 칼마비율 = CAGR / MDD                         | 높을수록  |
| 상대성과 | alpha              | 젠센 알파 (벤치마크 대비 OLS 절편, 연환산)     | 높을수록  |
|          | beta               | 벤치마크 대비 베타 (OLS 기울기)               | 1 근처    |
|          | information_ratio  | 벤치마크 대비 정보비율 (연환산)               | 높을수록  |

[보조 지표] 명세 외에 리포트·분석에 쓰는 값
| win_rate     | 일간 수익률 > 0 인 날의 비율 |
| avg_turnover | 평균 일간 회전율 (거래비용의 원인) |

상대성과 3개는 benchmark 를 줘야 계산된다 (없으면 NaN).
evaluate.py 는 기본 벤치마크로 SPY(S&P500 추종 ETF)를 쓴다. 지수 원본이 없어서 쓰는 대용이며,
팀이 다른 벤치마크(069500=KOSPI200 등)로 정하면 --benchmark 로 바꾼다.
evaluation-xai metrics.py 도 같은 정의(OLS 알파·베타, 벤치마크 대비 IR)를 쓰므로 수치를 맞춰볼 수 있다.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252                         # 연환산 기준 거래일 수

# 보고서·CSV 컬럼 순서를 고정하기 위한 명세 12개 지표 이름
METRIC_NAMES = [
    "cum_return", "cagr",                                  # 수익률
    "ann_vol", "var_95", "cvar_95", "max_mdd",             # 리스크
    "sharpe", "sortino", "calmar",                         # 위험조정
    "alpha", "beta", "information_ratio",                  # 상대성과 (벤치마크 대비)
]
EXTRA_METRIC_NAMES = ["win_rate", "avg_turnover"]          # 보조 지표
ALL_METRIC_NAMES = METRIC_NAMES + EXTRA_METRIC_NAMES


def max_drawdown(returns) -> float:
    """일간 수익률 시계열의 최대낙폭. 시작 자산(1.0)도 고점 후보에 넣는다."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return np.nan
    value = np.concatenate([[1.0], np.cumprod(1.0 + r)])   # 자산가치 곡선 (시작=1)
    peak = np.maximum.accumulate(value)                      # 그 시점까지의 최고점
    return float(((peak - value) / peak).max())


def compute_metrics(returns, turnover=None, benchmark=None, rf: float = 0.0,
                    var_level: float = 0.95, periods: int = TRADING_DAYS) -> dict:
    """명세 12개 + 보조 2개 성과지표를 dict로 돌려준다.

    returns  : 비용 차감 후 일간 단순수익률 (pd.Series 권장, index=날짜)
    turnover : 같은 날짜의 일간 회전율 (없으면 avg_turnover=NaN)
    benchmark: 같은 날짜의 벤치마크 일간 단순수익률 (없으면 alpha/beta/information_ratio=NaN)
               returns 와 index(날짜)로 맞춘다. index가 없는 배열이면 순서대로 맞춘다.
    rf       : 연 무위험수익률 (기본 0: rollout.summarize 와 같은 기준)
    """
    r = pd.Series(returns, dtype=np.float64).dropna()
    n = len(r)
    nan_result = {k: np.nan for k in ALL_METRIC_NAMES}
    if n < 2:
        return nan_result

    rf_daily = (1.0 + rf) ** (1.0 / periods) - 1.0           # 연 무위험수익률 -> 일간
    excess = r - rf_daily

    cum_return = float(np.prod(1.0 + r) - 1.0)
    cagr = float((1.0 + cum_return) ** (periods / n) - 1.0) if cum_return > -1.0 else -1.0
    std = float(r.std(ddof=1))
    ann_vol = std * np.sqrt(periods)
    sharpe = float(excess.mean() / std * np.sqrt(periods)) if std > 0 else np.nan

    # 소르티노: 목표수익률(rf) 아래로 빠진 부분만으로 하방 편차를 잰다 (0 이상인 날은 0으로)
    downside = np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))
    sortino = float(excess.mean() / downside * np.sqrt(periods)) if downside > 0 else np.nan

    mdd = max_drawdown(r)
    calmar = float(cagr / mdd) if mdd > 0 else np.nan

    # VaR/CVaR: 하위 (1-var_level) 분위수 손실. 손실을 양수로 표시한다.
    q = float(np.quantile(r, 1.0 - var_level))
    var = -q
    tail = r[r <= q]
    cvar = float(-tail.mean()) if len(tail) > 0 else np.nan

    avg_turnover = np.nan
    if turnover is not None:
        avg_turnover = float(pd.Series(turnover, dtype=np.float64).mean())

    alpha = beta = information_ratio = np.nan
    if benchmark is not None:
        b = benchmark if isinstance(benchmark, pd.Series) else pd.Series(benchmark, index=r.index)
        b = pd.Series(b, dtype=np.float64).reindex(r.index)  # 같은 날짜끼리만 비교
        both = pd.concat([r, b], axis=1, keys=["r", "b"]).dropna()
        if len(both) > 2:
            r_ex = both["r"] - rf_daily
            b_ex = both["b"] - rf_daily
            b_var = float(b_ex.var(ddof=1))
            if b_var > 1e-16:
                # 젠센 알파/베타: r_ex = alpha + beta * b_ex + e  (최소제곱)
                beta = float(np.cov(r_ex, b_ex, ddof=1)[0, 1] / b_var)
                alpha = float((r_ex.mean() - beta * b_ex.mean()) * periods)   # 일간 절편 x 252 (연환산)
            active = both["r"] - both["b"]
            a_std = float(active.std(ddof=1))
            if a_std > 1e-12:                                 # 벤치마크 자기 자신이면 정의 안 됨
                information_ratio = float(active.mean() / a_std * np.sqrt(periods))

    return {
        "cum_return": cum_return,
        "cagr": cagr,
        "ann_vol": float(ann_vol),
        "var_95": float(var),
        "cvar_95": cvar,
        "max_mdd": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "win_rate": float((r > 0).mean()),
        "avg_turnover": avg_turnover,
    }
