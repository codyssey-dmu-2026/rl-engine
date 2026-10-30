"""환경·데이터 로더·모델 메타 검증 (pytest).

실행 (프로젝트 루트에서):  pytest
         또는 src 에서:     python test_environment.py

가짜 데이터는 테스트마다 임시 폴더에 새로 만든다 (data/dummy 를 건드리지 않음).
실데이터(data/processed)가 있으면 실데이터 전용 테스트도 함께 돈다 (없으면 skip).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data_loader
import make_dummy_data
from data_loader import load_walk_forward_window, resolve_data_dir
from environment import (MAX_ASSET_WEIGHT, REWARD_TYPES, PortfolioEnv, action_to_weights,
                         cap_weights, weights_to_action)
from model_meta import build_meta, check_compatible

try:
    from stable_baselines3.common.env_checker import check_env   # SB3 규격 검사 (권장)
except ImportError:
    from gymnasium.utils.env_checker import check_env            # SB3가 없으면 gymnasium 기본 검사

REAL_DIR = data_loader.DEFAULT_DATA_DIR
HAS_REAL = (REAL_DIR / "walk_forward_manifest.json").exists()
GAMMA = 0.99


# ---------------------------------------------------------------------------
# 준비물 (fixture)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dummy_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("dummy")
    make_dummy_data.main(seed=0, out_dir=out)
    return out


@pytest.fixture(scope="session")
def data(dummy_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # '가짜 데이터' 경고는 여기선 정상
        return load_walk_forward_window(1, dummy_dir)


def make_env(data, **kw):
    return PortfolioEnv(data.train_features, data.train_returns, **kw)


# ---------------------------------------------------------------------------
# 1. 데이터 로더
# ---------------------------------------------------------------------------
def test_split_is_time_ordered(data):
    """fit < val < test 순서, 겹침 없음, fit+val = train, features/returns 날짜 일치."""
    assert data.fit_features.index.max() < data.val_features.index.min()
    assert data.val_features.index.max() < data.test_features.index.min()
    assert len(data.fit_features) + len(data.val_features) == len(data.train_features)
    assert data.train_features.index.equals(data.train_returns.index)
    assert data.is_dummy, "가짜 데이터인데 is_dummy 표시가 없음"


def test_no_silent_fallback_to_dummy(monkeypatch, tmp_path):
    """실데이터가 없으면 가짜로 조용히 바꾸지 않고 에러를 낸다."""
    monkeypatch.setattr(data_loader, "DEFAULT_DATA_DIR", tmp_path / "없는폴더")
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(None)
    assert resolve_data_dir(str(tmp_path)) == Path(tmp_path)    # 직접 지정하면 그대로 쓴다


def test_dummy_ticker_order_matches_real(data):
    """가짜 데이터의 자산 순서 = 실데이터 순서 (문자열 정렬). 다르면 모델을 섞어 쓸 수 없다."""
    assert data.tickers == sorted(data.tickers) == make_dummy_data.TICKERS
    if HAS_REAL:
        real = load_walk_forward_window(1, REAL_DIR)
        assert real.tickers == data.tickers
        assert list(real.train_features.columns) == list(data.train_features.columns)


def test_risk_file_adds_observation(data, dummy_dir, tmp_path):
    """RiskTag 파일을 주면 {ticker}_risk 12개가 관측에 붙어 288 -> 300차원이 된다."""
    base_dim = make_env(data).observation_space.shape[0]
    idx = data.train_features.index.append(data.test_features.index)
    risk = pd.DataFrame(0.0, index=idx[:-5],                      # 마지막 5일은 일부러 비움 -> 0으로 채움
                        columns=[f"{t}_risk" for t in data.tickers])
    path = tmp_path / "risk.parquet"
    risk.to_parquet(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = load_walk_forward_window(1, dummy_dir, risk_file=path)
    assert any("RiskTag가 없는 날짜" in str(w.message) for w in caught)
    assert d.risk_columns == [f"{t}_risk" for t in d.tickers]
    assert make_env(d).observation_space.shape[0] == base_dim + len(d.tickers)

    risk.drop(columns=[f"{d.tickers[0]}_risk"]).to_parquet(path)  # 컬럼이 빠지면 에러
    with pytest.raises(ValueError):
        load_walk_forward_window(1, dummy_dir, risk_file=path)


# ---------------------------------------------------------------------------
# 2. 환경 규격과 기본 동작
# ---------------------------------------------------------------------------
def test_gym_api_compliance(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                          # 행동공간 [-10,10] 권고 경고는 의도된 설계
        check_env(make_env(data))


def test_random_agent_episode(data):
    """랜덤 행동으로 끝까지: NaN 없음, 비중 합=1, 목표 비중 <= 0.4."""
    env = make_env(data)
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    while True:
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert abs(info["weights"].sum() - 1.0) < 1e-4
        assert abs(info["target_weights"].sum() - 1.0) < 1e-9
        assert info["target_weights"].max() <= MAX_ASSET_WEIGHT
        if term or trunc:
            break


@pytest.mark.parametrize("reward_type", REWARD_TYPES)
def test_reward_types_finite(data, reward_type):
    env = make_env(data, reward_type=reward_type)
    env.reset(seed=0)
    for _ in range(100):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if term or trunc:
            break


def test_safeguard_fires(data):
    """한도를 0.1%로 낮추면 조기 종료(terminated)가 실제로 일어난다."""
    env = make_env(data, mdd_limit=0.001)
    env.reset(seed=1)
    for _ in range(len(data.train_features)):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        if term or trunc:
            break
    assert term and info["drawdown"] > 0.001


def test_random_start_for_training_only(data):
    """학습용은 시작일이 고르게 바뀌고, 평가용은 항상 같은 날 시작한다."""
    rs = make_env(data, random_start=True)
    starts = set()
    for seed in range(40):
        rs.reset(seed=seed)
        starts.add(rs.current_step)
        assert rs.current_step <= rs.n_steps - 1 - rs.min_episode_days
    assert len(starts) > 20
    fixed = make_env(data)
    fixed.reset(seed=0)
    a = fixed.current_step
    fixed.reset(seed=99)
    assert fixed.current_step == a


# ---------------------------------------------------------------------------
# 3. 비중 변환 (상한 0.4, 배치 입력)
# ---------------------------------------------------------------------------
def test_weight_cap_on_extreme_actions():
    rng = np.random.default_rng(0)
    all_in = np.full(12, -10.0)
    all_in[0] = 10.0                                             # 한 자산에 몰빵하려는 행동
    for a in [all_in, np.zeros(12)] + [rng.normal(0, 5, 12) for _ in range(500)]:
        w = action_to_weights(a)
        assert abs(w.sum() - 1.0) < 1e-9 and w.min() >= 0 and w.max() <= MAX_ASSET_WEIGHT
    assert np.allclose(action_to_weights(np.zeros(12)), 1 / 12)  # 0 행동 = 동일가중


def test_cap_weights_known_values():
    # 넘친 0.3을 0.2:0.1로 나누면 두 번째도 넘쳐서 한 번 더 -> 0.4, 0.4, 0.2
    assert np.allclose(cap_weights(np.array([0.7, 0.2, 0.1]), 0.4), [0.4, 0.4, 0.2])
    with pytest.raises(ValueError):
        cap_weights(np.array([0.5, 0.5]), 0.4)                   # 2개 x 0.4 < 1 이라 불가능


def test_batch_actions_for_shap():
    """evaluation-xai 어댑터처럼 (batch, N) 행동을 넘겨도 행마다 같은 결과가 나온다."""
    rng = np.random.default_rng(3)
    for batch in (1, 3, 64):
        a = rng.normal(0, 5, (batch, 12)).astype(np.float32)
        w = action_to_weights(a)
        assert w.shape == (batch, 12)
        assert np.allclose(w, np.stack([action_to_weights(row) for row in a]))
        assert np.allclose(w.sum(axis=1), 1.0) and w.max() <= MAX_ASSET_WEIGHT
        assert np.allclose(action_to_weights(weights_to_action(w)), w, atol=1e-6)
    with pytest.raises(ValueError):
        action_to_weights(np.zeros((2, 3, 12)))


# ---------------------------------------------------------------------------
# 4. MDD 보상: '일부러 15% 깨고 빨리 끝내기'가 이득이 아닌지
# ---------------------------------------------------------------------------
def discounted_return(env, policy, gamma=GAMMA):
    """같은 시작일에서 정책을 끝까지 돌린 할인 누적 보상 (PPO가 최대화하는 값)."""
    env.reset(seed=0)
    total, discount = 0.0, 1.0
    while True:
        _, r, term, trunc, _ = env.step(policy(env))
        total += discount * r
        discount *= gamma
        if term or trunc:
            return total, term


def crash_policy(env):
    """다음 날 가장 많이 떨어질 3개 자산에 상한까지 몰아 일부러 낙폭을 만든다 (미래를 보는 반칙 정책).

    보상 구조에 빈틈이 있는지 보는 용도로는 가장 강한 공격이다.
    """
    nxt = env.simple_returns[env.current_step + 1]
    a = np.full(env.n_assets, -10.0)
    a[np.argsort(nxt)[:3]] = 10.0
    return a


def equal_weight(env):
    return np.zeros(env.n_assets)


def check_no_early_exit_incentive(features, returns, lam):
    kw = dict(reward_type="mdd", mdd_lambda=lam, gamma=GAMMA)
    ew, _ = discounted_return(PortfolioEnv(features, returns, **kw), equal_weight)
    cr, cr_term = discounted_return(PortfolioEnv(features, returns, **kw), crash_policy)
    assert cr_term, "반칙 정책이 15%를 못 깸 (테스트 설정 확인)"
    assert cr < ew, f"lambda={lam}: 조기 종료가 이득 (몰빵 {cr:.2f} >= 동일가중 {ew:.2f})"


@pytest.mark.parametrize("lam", [0.5, 1.0, 5.0])
def test_early_exit_not_profitable_dummy(data, lam):
    check_no_early_exit_incentive(data.train_features, data.train_returns, lam)
    check_no_early_exit_incentive(data.test_features, data.test_returns, lam)


@pytest.mark.skipif(not HAS_REAL, reason="data/processed 실데이터 없음")
@pytest.mark.parametrize("window", [1, 2, 3, 4])
@pytest.mark.parametrize("lam", [0.5, 1.0, 5.0])
def test_early_exit_not_profitable_real(window, lam):
    d = load_walk_forward_window(window, REAL_DIR)
    check_no_early_exit_incentive(d.train_features, d.train_returns, lam)


def test_safeguard_penalty_is_bounded(data):
    """할인 반영 페널티는 lambda * 0.15 / (1 - gamma) 를 넘지 않는다 (lambda=5 -> 75 이하)."""
    env = make_env(data, reward_type="mdd", mdd_lambda=5.0, gamma=GAMMA)
    env.reset(seed=0)
    env.current_step = 0                                         # 남은 스텝이 가장 많은 경우
    assert env._safeguard_penalty() <= 5.0 * 0.15 / (1 - GAMMA) + 1e-9
    assert make_env(data, reward_type="simple")._safeguard_penalty() == 0.0


# ---------------------------------------------------------------------------
# 5. 모델 메타 정보 (다른 데이터로 학습한 모델이 섞이는 것 방지)
# ---------------------------------------------------------------------------
def test_model_meta_compatibility(data):
    env = make_env(data)
    meta = build_meta(data, env, stage="test", name="simple", reward_type="simple", mdd_lambda=1.0,
                      seed=42, timesteps=100, gamma=GAMMA, data_dir="dummy")
    check_compatible(meta, data, env, allow_dummy=True)          # 같은 데이터면 통과
    with pytest.raises(ValueError, match="가짜"):
        check_compatible(meta, data, env)                        # 가짜 데이터 모델은 기본적으로 거부
    swapped = dict(meta, tickers=meta["tickers"][::-1])
    with pytest.raises(ValueError, match="자산 순서"):
        check_compatible(swapped, data, env, allow_dummy=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
