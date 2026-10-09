import argparse                            # 실행할 때 데이터 폴더를 바꿀 수 있게 함
from pathlib import Path                   # 경로 처리

import numpy as np                         # 수치 검증

from data_loader import load_walk_forward_window   # 데이터 로더
from environment import PortfolioEnv, REWARD_TYPES  # 강화학습 환경

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
        if terminated or truncated:
            break
    print(f"[4] 랜덤 에이전트 {steps}스텝 | 종료 사유: "
          f"{'Safe-Guard(MDD 초과)' if terminated else '데이터 끝'} | "
          f"최종 자산가치 {info['portfolio_value']:,.0f} | MDD {info['mdd']:.2%}")

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

    print("\n모든 검증 통과")


if __name__ == "__main__":
    main()