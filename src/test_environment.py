import argparse                            # 실행할 때 데이터 폴더를 바꿀 수 있게 함
from pathlib import Path                   # 경로 처리

import numpy as np                         # 수치 검증

from data_loader import load_walk_forward_window   # 데이터 로더
from environment import MAX_ASSET_WEIGHT, PortfolioEnv, REWARD_TYPES, action_to_weights, cap_weights  # 강화학습 환경

try:
    from stable_baselines3.common.env_checker import check_env   # SB3 규격 검사 (권장)
except ImportError:
    from gymnasium.utils.env_checker import check_env            # SB3가 없으면 gymnasium 기본 검사

ROOT = Path(__file__).resolve().parent.parent       # 프로젝트 루트


def pick_default_dir():
    """실제 데이터(data/processed)가 있으면 그걸, 없으면 가짜 데이터(data/dummy)를 쓴다."""
    real = ROOT / "data" / "processed"
    return real if (real / "walk_forward_manifest.json").exists() else ROOT / "data" / "dummy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="parquet 파일이 있는 폴더")
    parser.add_argument("--window", type=int, default=1, help="walk-forward 윈도우 번호")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else pick_default_dir()
    print(f"[1] 데이터 폴더: {data_dir}")

    # ---------- 1. 데이터 로드 + 시계열 분리 확인 ----------
    data = load_walk_forward_window(args.window, data_dir)
    print(f"[2] window {data.window}: 학습 {len(data.train_features)}일 "
          f"({data.train_features.index.min().date()} ~ {data.train_features.index.max().date()}), "
          f"테스트 {len(data.test_features)}일 "
          f"({data.test_features.index.min().date()} ~ {data.test_features.index.max().date()})")
    assert data.train_features.index.max() < data.test_features.index.min(), "학습/테스트 날짜가 겹침!"
    assert data.train_features.index.equals(data.train_returns.index), "features/returns 날짜 불일치!"
    # 검증 구간: 학습 구간 마지막 6개월, fit < val < test 순서, fit+val = train
    assert data.fit_features.index.max() < data.val_features.index.min(), "fit/val 날짜가 겹침!"
    assert data.val_features.index.max() < data.test_features.index.min(), "val/test 날짜가 겹침!"
    assert len(data.fit_features) + len(data.val_features) == len(data.train_features), "fit+val != train!"
    print(f"    검증 구간 {len(data.val_features)}일 "
          f"({data.val_features.index.min().date()} ~ {data.val_features.index.max().date()}) "
          f"| fit {len(data.fit_features)}일")
    print(f"    피처 컬럼 {data.train_features.shape[1]}개, 수익률 컬럼 {data.train_returns.shape[1]}개")

    # ---------- 2. Gymnasium 규격 검사 ----------
    env = PortfolioEnv(data.train_features, data.train_returns, reward_type="simple")
    check_env(env)
    print(f"[3] 환경 규격 검증 통과 | 자산 {env.n_assets}개 | 관측 차원 {env.observation_space.shape[0]}")

    # ---------- 3. 랜덤 에이전트로 끝까지 돌려보기 ----------
    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape, "관측 shape 불일치!"
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        assert np.isfinite(obs).all() and np.isfinite(reward), "NaN/inf 발생!"
        assert abs(info["weights"].sum() - 1.0) < 1e-4, "비중합이 1이 아님!"
        assert abs(info["target_weights"].sum() - 1.0) < 1e-4, "목표 비중합이 1이 아님!"
        assert info["target_weights"].max() <= MAX_ASSET_WEIGHT, "목표 비중이 상한 0.4를 넘음!"
        if terminated or truncated:
            break
    print(f"[4] 랜덤 에이전트 {steps}스텝 | 종료 사유: "
          f"{'Safe-Guard(MDD 초과)' if terminated else '데이터 끝'} | "
          f"최종 자산가치 {info['portfolio_value']:,.0f} | 현재 낙폭 {info['drawdown']:.2%}")

    # ---------- 3-1. 비중 상한 0.4: 극단적인 행동에서도 지켜지는지 ----------
    rng = np.random.default_rng(0)
    extreme = [np.full(env.n_assets, -10.0), np.zeros(env.n_assets)]
    extreme[0][0] = 10.0                                             # 한 자산에 몰빵하려는 행동
    extreme += [rng.normal(0, 5, env.n_assets) for _ in range(500)]  # 무작위 큰 행동
    for a in extreme:
        w = action_to_weights(a)
        assert abs(w.sum() - 1.0) < 1e-9 and w.min() >= 0, "상한 투영 후 합/부호 오류!"
        assert w.max() <= MAX_ASSET_WEIGHT, f"상한 초과: {w.max()}"
    w = cap_weights(np.array([0.7, 0.2, 0.1]), 0.4)                  # 손계산: 넘친 0.3을 0.2:0.1로 나눔
    assert np.allclose(w, [0.4, 0.4, 0.2]), w                        # -> 0.4, 0.4, 0.2 (두 번째도 넘쳐서 반복)
    assert np.allclose(action_to_weights(np.zeros(env.n_assets)), 1 / env.n_assets), "동일가중이 깨짐!"
    print(f"[4-1] 비중 상한 확인: 한 자산 몰빵 행동 -> 최대 비중 {action_to_weights(extreme[0]).max():.3f}")

    # ---------- 4. 보상 함수 3종이 모두 정상 동작하는지 ----------
    for reward_type in REWARD_TYPES:
        e = PortfolioEnv(data.train_features, data.train_returns, reward_type=reward_type)
        e.reset(seed=0)
        rewards = []
        for _ in range(100):
            _, r, term, trunc, _ = e.step(e.action_space.sample())
            rewards.append(r)
            if term or trunc:
                break
        assert np.isfinite(rewards).all(), f"{reward_type} 보상에 NaN/inf!"
        print(f"[5] 보상 '{reward_type}': {len(rewards)}스텝 평균 {np.mean(rewards):+.5f}, "
              f"최대 {np.max(rewards):+.5f}, 최소 {np.min(rewards):+.5f}")

    # ---------- 5. Safe-Guard가 실제로 발동하는지 (한계를 일부러 매우 낮춤) ----------
    strict = PortfolioEnv(data.train_features, data.train_returns, mdd_limit=0.001)
    strict.reset(seed=1)
    fired = False
    for _ in range(len(data.train_features)):
        _, _, term, trunc, _ = strict.step(strict.action_space.sample())
        if term:
            fired = True
            break
        if trunc:
            break
    assert fired, "mdd_limit=0.1%인데 조기 종료가 안 됨!"
    print("[6] Safe-Guard 발동 확인 (mdd_limit=0.1%에서 조기 종료)")

    # ---------- 6. 무작위 시작일(학습용)이 실제로 구간 전체를 고르게 쓰는지 ----------
    rs = PortfolioEnv(data.train_features, data.train_returns, random_start=True)
    starts = set()
    for seed in range(40):
        rs.reset(seed=seed)
        starts.add(rs.current_step)
        assert rs.current_step <= rs.n_steps - 1 - rs.min_episode_days, "시작일이 너무 늦음!"
    assert len(starts) > 20, "시작일이 거의 안 바뀜!"
    fixed = PortfolioEnv(data.train_features, data.train_returns)           # 평가용은 고정 시작
    fixed.reset(seed=0); a = fixed.current_step
    fixed.reset(seed=99); assert fixed.current_step == a, "평가 환경 시작일이 흔들림!"
    print(f"[7] 무작위 시작 확인: 40회 중 서로 다른 시작일 {len(starts)}개 | 평가 환경은 고정 시작")

    # ---------- 7. MDD 보상에서 '일부러 15% 깨고 빨리 끝내기'가 이득이 아닌지 ----------
    # 리뷰 재현: lambda=1, 같은 시작일에서 (a) 동일가중을 끝까지 vs (b) 매일 다음 날 가장 많이 떨어질
    # 자산에 상한(0.4)까지 몰아 일부러 15% 낙폭을 만드는 정책. (b)는 미래를 보는 '반칙' 정책이지만
    # 보상 구조에 빈틈이 있는지 보는 용도로는 가장 강한 공격이다.
    def total_reward(env, policy):
        env.reset(seed=0)
        total, steps = 0.0, 0
        while True:
            _, r, term, trunc, info = env.step(policy(env))
            total, steps = total + r, steps + 1
            if term or trunc:
                return total, steps, term

    def crash_policy(e):
        nxt = e.simple_returns[e.current_step + 1]                   # 다음 날 수익률 (반칙)
        a = np.full(e.n_assets, -10.0)
        a[np.argsort(nxt)[:3]] = 10.0                                # 가장 나쁜 3개에 몰기 (상한 때문에 3개 필요)
        return a

    for features, returns, label in ((data.train_features, data.train_returns, "train"),
                                     (data.test_features, data.test_returns, "test")):
        ew_env = PortfolioEnv(features, returns, reward_type="mdd", mdd_lambda=1.0)
        cr_env = PortfolioEnv(features, returns, reward_type="mdd", mdd_lambda=1.0)
        ew, ew_steps, _ = total_reward(ew_env, lambda e: np.zeros(e.n_assets))
        cr, cr_steps, cr_term = total_reward(cr_env, crash_policy)
        assert cr_term, "몰빵 정책이 15%를 못 깸 (테스트 설정 확인 필요)"
        assert cr < ew, f"[{label}] 조기 종료가 여전히 이득: 몰빵 {cr:.1f} >= 동일가중 {ew:.1f}"
        print(f"[8] MDD 보상 조기종료 확인({label}, lambda=1): 동일가중 {ew_steps}스텝 보상합 {ew:+.1f} "
              f"> 15% 몰빵 {cr_steps}스텝 종료 보상합 {cr:+.1f}")

    print("\n모든 검증 통과")


if __name__ == "__main__":
    main()