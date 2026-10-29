import argparse                            # 실행할 때 옵션을 받기 위함
from pathlib import Path                   # 경로 처리

import numpy as np                         # 보상 클리핑 해제(np.inf)용

from stable_baselines3 import PPO                                    # PPO 알고리즘
from stable_baselines3.common.monitor import Monitor                 # 에피소드 보상 기록 (학습곡선용)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # 벡터 환경 + 보상 스케일 보정

from data_loader import DEFAULT_DATA_DIR, load_walk_forward_window   # 데이터 로더
from environment import PortfolioEnv                                  # 강화학습 환경
from rollout import build_test_env, build_val_env, equal_weight_action_fn, run_episode, summarize

ROOT = Path(__file__).resolve().parent.parent      # 프로젝트 루트
MODELS_DIR = ROOT / "models"                       # 모델 저장 폴더
LOGS_DIR = ROOT / "logs"                           # tensorboard 로그 폴더

# stage 별로 '어디서 학습하고 어디서 평가하는지'
#   val : fit 구간에서 학습 -> val 구간에서 평가  (보상형/lambda/하이퍼파라미터 선택은 여기서만)
#   test: train 전체(fit+val)에서 재학습 -> test 구간에서 평가  (고른 설정의 최종 성과 보고)
STAGES = ("val", "test")


def split_for_stage(data, stage):
    """stage에 맞는 (학습 피처, 학습 수익률)을 돌려준다."""
    if stage == "val":
        return data.fit_features, data.fit_returns
    if stage == "test":
        return data.train_features, data.train_returns
    raise ValueError(f"stage는 {STAGES} 중 하나여야 합니다: {stage}")


def build_eval_env_for_stage(data, stage, lookback):
    """stage에 맞는 평가 환경 (val 또는 test 구간)."""
    return build_val_env(data, lookback) if stage == "val" else build_test_env(data, lookback)


def pick_data_dir(cli_value):
    """--data-dir 이 없으면 실제 데이터(data/processed), 그것도 없으면 가짜(data/dummy)를 쓴다."""
    if cli_value:
        return Path(cli_value)
    if (DEFAULT_DATA_DIR / "walk_forward_manifest.json").exists():
        return DEFAULT_DATA_DIR
    return ROOT / "data" / "dummy"


def train_one(data, name, reward_type, mdd_lambda, seed, timesteps, lookback, stage="val"):
    """보상함수 한 가지로 PPO를 학습하고, stage에 맞는 평가 구간 성과를 dict로 돌려준다."""
    run_name = f"ppo_{stage}_w{data.window}_{name}_s{seed}"               # 예: ppo_val_w1_mdd_l1.0_s42
    MODELS_DIR.mkdir(exist_ok=True)                                       # 폴더 없으면 생성
    LOGS_DIR.mkdir(exist_ok=True)
    fit_features, fit_returns = split_for_stage(data, stage)

    # ---------- 학습 환경: 평가 구간 이전 데이터만 사용 (평가 구간은 절대 안 봄) ----------
    train_env = Monitor(PortfolioEnv(
        fit_features, fit_returns,
        lookback=lookback, reward_type=reward_type, mdd_lambda=mdd_lambda,
        random_start=True,                 # 학습 때만: 에피소드마다 시작일을 무작위로 (평가는 고정 시작)
    ))
    # 보상 스케일 보정: 보상함수마다 값의 크기가 수십~수백 배 달라서(예: sharpe vs simple)
    # PPO 학습이 불안정해지는 것을 막는다. 관측값은 이미 정규화돼 있으므로 norm_obs=False.
    # clip_reward=np.inf: 기본값(10)이면 Safe-Guard 종료 페널티가 잘려서 '빨리 끝내기' 방지 효과가 사라진다.
    vec_env = VecNormalize(DummyVecEnv([lambda: train_env]), norm_obs=False, norm_reward=True,
                           clip_reward=np.inf)

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
    # 모델 저장. '.zip'을 직접 붙인다: 'mdd_l1.0_s42' 처럼 이름에 점이 있으면 SB3가
    # '.0_s42'를 확장자로 보고 .zip을 안 붙여서, 변형마다 파일 이름 규칙이 달라지기 때문.
    model.save(str(MODELS_DIR / f"{run_name}.zip"))

    # ---------- 평가 구간: 학습된 정책을 '결정론적'으로 한 번 돌린다 ----------
    eval_env = build_eval_env_for_stage(data, stage, lookback)
    df = run_episode(eval_env, lambda obs: model.predict(obs, deterministic=True)[0])
    metrics = summarize(df)
    metrics.update({"stage": stage, "strategy": f"PPO-{name}", "reward_type": reward_type,
                    "mdd_lambda": mdd_lambda, "window": data.window, "seed": seed, "timesteps": timesteps})
    return metrics


def baseline_metrics(data, lookback, stage="val"):
    """같은 평가 구간·같은 거래비용에서 동일가중 포트폴리오 성과 (비교 기준선)."""
    eval_env = build_eval_env_for_stage(data, stage, lookback)
    df = run_episode(eval_env, equal_weight_action_fn(eval_env.n_assets))
    metrics = summarize(df)
    metrics.update({"stage": stage, "strategy": "EqualWeight", "reward_type": "-", "mdd_lambda": np.nan,
                    "window": data.window, "seed": "-", "timesteps": 0})
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
    parser.add_argument("--stage", default="val", choices=STAGES,
                        help="val: fit 학습 -> val 평가(선택용, 기본) / test: 전체 재학습 -> test 평가(최종 보고용)")
    args = parser.parse_args()

    data_dir = pick_data_dir(args.data_dir)
    print(f"데이터 폴더: {data_dir}")
    data = load_walk_forward_window(args.window, data_dir)
    print(f"window {data.window}: fit {len(data.fit_features)}일 / val {len(data.val_features)}일 "
          f"/ test {len(data.test_features)}일")

    name = args.reward if args.reward != "mdd" else f"mdd_l{args.mdd_lambda}"
    print(f"학습 시작: {name}, {args.timesteps} 스텝, seed={args.seed}, stage={args.stage}")
    ppo = train_one(data, name, args.reward, args.mdd_lambda, args.seed, args.timesteps, args.lookback,
                    stage=args.stage)
    base = baseline_metrics(data, args.lookback, stage=args.stage)

    print(f"\n=== {args.stage} 구간 성과 비교 ===")
    for m in (ppo, base):
        print(f"{m['strategy']:>16} | 누적수익률 {m['cum_return']:+.2%} | 샤프 {m['sharpe']:+.2f} | "
              f"MDD {m['max_mdd']:.2%} | 평균회전율 {m['avg_turnover']:.3f} | 최대비중 {m['max_weight']:.3f}")


if __name__ == "__main__":
    main()