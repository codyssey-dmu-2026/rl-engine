import argparse                            # 실행할 때 옵션을 받기 위함
from pathlib import Path                   # 경로 처리

from stable_baselines3 import PPO                                    # PPO 알고리즘
from stable_baselines3.common.monitor import Monitor                 # 에피소드 보상 기록 (학습곡선용)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # 벡터 환경 + 보상 스케일 보정

from data_loader import DEFAULT_DATA_DIR, load_walk_forward_window   # 데이터 로더
from environment import PortfolioEnv                                  # 강화학습 환경
from rollout import build_test_env, equal_weight_action_fn, run_episode, summarize

ROOT = Path(__file__).resolve().parent.parent      # 프로젝트 루트
MODELS_DIR = ROOT / "models"                       # 모델 저장 폴더
LOGS_DIR = ROOT / "logs"                           # tensorboard 로그 폴더


def pick_data_dir(cli_value):
    """--data-dir 이 없으면 실제 데이터(data/processed), 그것도 없으면 가짜(data/dummy)를 쓴다."""
    if cli_value:
        return Path(cli_value)
    if (DEFAULT_DATA_DIR / "walk_forward_manifest.json").exists():
        return DEFAULT_DATA_DIR
    return ROOT / "data" / "dummy"


def train_one(data, name, reward_type, mdd_lambda, seed, timesteps, lookback):
    """보상함수 한 가지로 PPO를 학습하고, 테스트 구간 성과를 dict로 돌려준다."""
    run_name = f"ppo_w{data.window}_{name}_s{seed}"                       # 예: ppo_w1_mdd_l1.0_s42
    MODELS_DIR.mkdir(exist_ok=True)                                       # 폴더 없으면 생성
    LOGS_DIR.mkdir(exist_ok=True)

    # ---------- 학습 환경: 학습 구간 데이터만 사용 (테스트 구간은 절대 안 봄) ----------
    train_env = Monitor(PortfolioEnv(
        data.train_features, data.train_returns,
        lookback=lookback, reward_type=reward_type, mdd_lambda=mdd_lambda,
        random_start=True,                 # 학습 때만: 에피소드마다 시작일을 무작위로 (평가는 고정 시작)
    ))
    # 보상 스케일 보정: 보상함수마다 값의 크기가 수십~수백 배 달라서(예: sharpe vs simple)
    # PPO 학습이 불안정해지는 것을 막는다. 관측값은 이미 정규화돼 있으므로 norm_obs=False.
    vec_env = VecNormalize(DummyVecEnv([lambda: train_env]), norm_obs=False, norm_reward=True)

    model = PPO(
        policy="MlpPolicy",                # 기본 다층 퍼셉트론 정책망
        env=vec_env,
        learning_rate=3e-4,                # 학습률
        n_steps=2048,                      # 업데이트 전 수집할 스텝 수
        batch_size=64,                     # 미니배치 크기
        n_epochs=10,                       # 수집 데이터 재사용 횟수
        gamma=0.99,                        # 할인율
        clip_range=0.2,                    # PPO 핵심: 정책 변화 제한폭
        seed=seed,                         # 시드 고정 (재현성)
        verbose=0,                         # 로그 최소화
        tensorboard_log=str(LOGS_DIR / run_name),  # 실행별 로그 폴더
    )
    model.learn(total_timesteps=timesteps)                                # 학습 (명세: 10만 스텝 이상)
    model.save(str(MODELS_DIR / run_name))                                # 모델 저장

    # ---------- 테스트 구간 평가: 학습된 정책을 '결정론적'으로 한 번 돌린다 ----------
    test_env = build_test_env(data, lookback)
    df = run_episode(test_env, lambda obs: model.predict(obs, deterministic=True)[0])
    metrics = summarize(df)
    metrics.update({"strategy": f"PPO-{name}", "window": data.window, "seed": seed, "timesteps": timesteps})
    return metrics


def baseline_metrics(data, lookback):
    """같은 테스트 구간·같은 거래비용에서 동일가중 포트폴리오 성과 (비교 기준선)."""
    test_env = build_test_env(data, lookback)
    df = run_episode(test_env, equal_weight_action_fn(test_env.n_assets))
    metrics = summarize(df)
    metrics.update({"strategy": "EqualWeight", "window": data.window, "seed": "-", "timesteps": 0})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="parquet 폴더 (기본: data/processed 또는 data/dummy)")
    parser.add_argument("--window", type=int, default=1, help="walk-forward 윈도우 번호 (1~4)")
    parser.add_argument("--reward", default="simple", choices=["simple", "sharpe", "mdd"])
    parser.add_argument("--mdd-lambda", type=float, default=1.0, help="mdd 보상의 페널티 강도 (0.5~5.0)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="학습 스텝 수 (처음엔 2000으로 시험)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lookback", type=int, default=20, help="과거 N일 수익률 윈도우 (20~60)")
    args = parser.parse_args()

    data_dir = pick_data_dir(args.data_dir)
    print(f"데이터 폴더: {data_dir}")
    data = load_walk_forward_window(args.window, data_dir)
    print(f"window {data.window}: 학습 {len(data.train_features)}일 / 테스트 {len(data.test_features)}일")

    name = args.reward if args.reward != "mdd" else f"mdd_l{args.mdd_lambda}"
    print(f"학습 시작: {name}, {args.timesteps} 스텝, seed={args.seed}")
    ppo = train_one(data, name, args.reward, args.mdd_lambda, args.seed, args.timesteps, args.lookback)
    base = baseline_metrics(data, args.lookback)

    print("\n=== 테스트 구간 성과 비교 ===")
    for m in (ppo, base):
        print(f"{m['strategy']:>16} | 누적수익률 {m['cum_return']:+.2%} | 샤프 {m['sharpe']:+.2f} | "
              f"MDD {m['max_mdd']:.2%} | 평균회전율 {m['avg_turnover']:.3f}")


if __name__ == "__main__":
    main()