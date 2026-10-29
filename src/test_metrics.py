"""metrics.py / weights_to_action 검증 (torch 없이 실행 가능).

실행: python test_metrics.py
"""
import numpy as np
import pandas as pd

from environment import action_to_weights, cap_weights, weights_to_action
from metrics import METRIC_NAMES, compute_metrics, max_drawdown
from rollout import summarize


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_known_values():
    # +10% 다음 -10%: 누적 = 1.1*0.9-1 = -1%, MDD = 10%
    m = compute_metrics(pd.Series([0.10, -0.10]), turnover=[0.5, 0.1])
    assert close(m["cum_return"], -0.01)
    assert close(m["max_mdd"], 0.10)
    assert close(m["win_rate"], 0.5)
    assert close(m["avg_turnover"], 0.3)
    assert set(m) == set(METRIC_NAMES) and len(METRIC_NAMES) == 12


def test_drawdown_counts_initial_capital():
    # 첫날부터 손실이면 시작 자본(1.0)이 고점이어야 한다
    assert close(max_drawdown([-0.05, -0.05]), 1 - 0.95 ** 2)
    assert close(max_drawdown([0.01, 0.02]), 0.0)


def test_var_cvar_are_positive_losses():
    r = pd.Series(np.linspace(-0.05, 0.05, 101))     # -5% ~ +5% 균등
    m = compute_metrics(r)
    assert m["var_95"] > 0 and m["cvar_95"] >= m["var_95"]
    assert close(m["var_95"], 0.045, 1e-6)              # 하위 5% 분위수


def test_information_ratio():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0.0003, 0.01, 300), index=idx)
    assert np.isnan(compute_metrics(bench, benchmark=bench)["information_ratio"])   # 자기 자신
    better = bench + 0.001 + rng.normal(0, 0.002, 300)
    assert compute_metrics(better, benchmark=bench)["information_ratio"] > 0


def test_matches_rollout_summarize():
    # 기존 summarize 와 겹치는 지표는 같은 값이어야 한다 (학습 로그와 평가표가 어긋나지 않게)
    idx = pd.bdate_range("2024-01-01", periods=250)
    r = np.random.default_rng(1).normal(0.0004, 0.01, 250)
    value = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], value]))[1:]
    df = pd.DataFrame({"net_return": r, "drawdown": (peak - value) / peak,
                       "turnover": 0.01, "max_target_weight": 0.4}, index=idx)
    s, m = summarize(df), compute_metrics(df["net_return"], df["turnover"])
    for k in ("cum_return", "ann_vol", "sharpe", "max_mdd", "avg_turnover"):
        assert close(s[k], m[k], 1e-9), k


def test_weights_to_action_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(100):
        w = cap_weights(rng.dirichlet(np.ones(12)))
        assert np.allclose(action_to_weights(weights_to_action(w)), w, atol=1e-6)
    # 일부 자산 0, 상한 꽉 찬 경우 (MVO에서 흔함)
    w = np.array([0.4, 0.4, 0.2] + [0.0] * 9)
    assert np.allclose(action_to_weights(weights_to_action(w)), w, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[통과] {t.__name__}")
    print(f"\n전체 {len(tests)}개 테스트 통과")
