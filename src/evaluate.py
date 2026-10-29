"""Walk-Forward 백테스트 + 12개 성과지표.

train_compare.py 가 저장한 PPO 모델(models/ppo_{stage}_w{n}_{변형}_s{seed}.zip)을 불러와
윈도우별 평가 구간(test 또는 val)을 결정론적으로 돌리고, 동일가중 기준선과 같은 지표로 비교한다.
학습은 하지 않는다 -> 평가만 여러 번 다시 돌려도 결과가 같다.

결과 (models/ 아래, 요약 CSV라 git 추적 대상):
  eval_{stage}_windows.csv     : 윈도우 x 전략 x 시드 별 12개 지표
  eval_{stage}_walkforward.csv : 윈도우 1~4 평가 구간을 이어붙인 전체 기간(out-of-sample) 지표
  eval_{stage}_summary.csv     : 전략별 walk-forward 지표의 시드 평균/표준편차 (리포트용)
  eval_{stage}_daily.csv       : 날짜별 수익률·낙폭·회전율 (그래프, ANOVA, evaluation-xai 연동용)

예)
  python train_compare.py --stage test --windows 1 2 3 4 --seeds 42 43 44   # 먼저 학습
  python evaluate.py --stage test --windows 1 2 3 4 --seeds 42 43 44        # 그다음 평가
  python evaluate.py --baselines-only                                        # 모델 없이 기준선만
"""
import argparse                            # 실행 옵션
from pathlib import Path                   # 경로 처리

import numpy as np                         # 수치 계산
import pandas as pd                        # 결과 표

from data_loader import DEFAULT_DATA_DIR, load_walk_forward_window
from metrics import METRIC_NAMES, compute_metrics
from rollout import build_test_env, build_val_env, equal_weight_action_fn, run_episode

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
STAGES = ("val", "test")
BENCHMARK = "EqualWeight"                  # 정보비율(information_ratio)의 기준 전략

# train_compare.EXPERIMENTS 와 같은 이름이어야 모델 파일을 찾는다.
# (train_compare를 import하면 torch까지 읽어 느려서, 이름만 여기 둔다. 변형을 바꾸면 둘 다 고칠 것)
PPO_VARIANTS = [
    {"name": "simple",   "reward_type": "simple", "mdd_lambda": np.nan},
    {"name": "sharpe",   "reward_type": "sharpe", "mdd_lambda": np.nan},
    {"name": "mdd_l0.5", "reward_type": "mdd",    "mdd_lambda": 0.5},
    {"name": "mdd_l1.0", "reward_type": "mdd",    "mdd_lambda": 1.0},
    {"name": "mdd_l5.0", "reward_type": "mdd",    "mdd_lambda": 5.0},
]


# ---------------------------------------------------------------------------
# 비교군(baseline) 등록부
#   이름 -> 함수(env, data, stage) -> action_fn(obs)
#   env를 받는 이유: MVO처럼 '오늘까지의 실제 수익률'이 필요한 전략은
#   env.simple_returns[: env.current_step + 1] 을 보고 비중을 정한 뒤
#   environment.weights_to_action(비중) 으로 행동을 돌려주면 된다. (미래 데이터 금지!)
# ---------------------------------------------------------------------------
BASELINES = {
    "EqualWeight": lambda env, data, stage: equal_weight_action_fn(env.n_assets),
    # "MVO": mvo_action_fn,   # TODO: MVO 비교군 구현 후 등록
}


def pick_data_dir(cli_value):
    """train.pick_data_dir 와 같은 규칙 (train.py는 torch를 읽어서 여기 따로 둔다)."""
    if cli_value:
        return Path(cli_value)
    if (DEFAULT_DATA_DIR / "walk_forward_manifest.json").exists():
        return DEFAULT_DATA_DIR
    return ROOT / "data" / "dummy"


def build_eval_env(data, stage, lookback):
    """stage에 맞는 평가 환경 (Safe-Guard 끔, 평가 구간 전체를 돈다)."""
    return build_val_env(data, lookback) if stage == "val" else build_test_env(data, lookback)


def model_path(models_dir, stage, window, variant, seed):
    """train.train_one 이 저장하는 이름 규칙과 동일.

    예전 train.py 는 'mdd_l1.0' 처럼 점이 들어간 이름을 .zip 없이 저장했다(SB3가 '.0_s42'를
    확장자로 착각). 이미 학습해 둔 모델도 쓸 수 있게 .zip 없는 이름도 찾아본다.
    """
    path = Path(models_dir) / f"ppo_{stage}_w{window}_{variant}_s{seed}.zip"
    legacy = path.with_name(path.stem)                       # .zip 뺀 이름
    return legacy if not path.exists() and legacy.is_file() else path


def load_ppo(path):
    """PPO 모델 로드 (torch는 여기서만 읽는다 -> 기준선만 돌릴 땐 필요 없음)."""
    from stable_baselines3 import PPO
    return PPO.load(str(path), device="cpu")


def run_strategy(data, stage, lookback, action_fn_factory):
    """전략 하나를 평가 구간에 돌려 날짜별 기록(DataFrame)을 돌려준다."""
    env = build_eval_env(data, stage, lookback)
    action_fn = action_fn_factory(env)
    return run_episode(env, action_fn)


def evaluate_window(data, stage, lookback, seeds, variants, models_dir, baselines_only=False):
    """한 윈도우의 모든 전략을 돌린다. (날짜별 기록 list, 없는 모델 경로 list)"""
    frames, missing = [], []

    def add(df, strategy, seed, reward_type, mdd_lambda):
        df = df.copy()
        df["window"] = data.window
        df["strategy"] = strategy
        df["seed"] = seed
        df["reward_type"] = reward_type
        df["mdd_lambda"] = mdd_lambda
        frames.append(df)

    for name, factory in BASELINES.items():
        df = run_strategy(data, stage, lookback, lambda env, f=factory: f(env, data, stage))
        add(df, name, "-", "-", np.nan)

    if baselines_only:
        return frames, missing

    for seed in seeds:
        for v in variants:
            path = model_path(models_dir, stage, data.window, v["name"], seed)
            if not path.exists():
                missing.append(path)
                continue
            model = load_ppo(path)
            df = run_strategy(data, stage, lookback,
                              lambda env, m=model: (lambda obs: m.predict(obs, deterministic=True)[0]))
            add(df, f"PPO-{v['name']}", seed, v["reward_type"], v["mdd_lambda"])
    return frames, missing


def metrics_table(daily, group_cols, mdd_limit=0.15):
    """날짜별 기록을 group_cols 단위로 묶어 12개 지표 표를 만든다.

    벤치마크(동일가중)는 같은 group에서 seed/strategy만 뺀 키로 찾는다.
    """
    bench_keys = [c for c in group_cols if c not in ("strategy", "seed", "reward_type", "mdd_lambda")]
    bench = daily[daily["strategy"] == BENCHMARK]
    bench_series = {k if isinstance(k, tuple) else (k,): g["net_return"]
                    for k, g in (bench.groupby(bench_keys) if bench_keys else [((), bench)])}

    rows = []
    for key, g in daily.groupby(group_cols, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        info = dict(zip(group_cols, key))
        g = g.sort_index()
        if g.index.has_duplicates:
            raise ValueError(f"{info}: 같은 날짜가 두 번 있습니다 (평가 구간이 겹침)")
        b = bench_series.get(tuple(info[c] for c in bench_keys))
        m = compute_metrics(g["net_return"], g["turnover"], benchmark=b)
        mdd = m["max_mdd"]
        rows.append({
            **info,
            "start": str(g.index.min().date()),
            "end": str(g.index.max().date()),
            "days": len(g),
            **m,
            "breach_15pct": bool(mdd > mdd_limit),                     # Safe-Guard 기준 초과 여부
            "max_weight": float(g["max_target_weight"].max()),         # <= 0.4 이어야 함
        })
    return pd.DataFrame(rows)


def summarize_over_seeds(walkforward):
    """전략별로 시드 평균/표준편차 (기준선은 시드가 없으니 std=NaN)."""
    agg = walkforward.groupby("strategy", sort=False)[METRIC_NAMES].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "n_seeds", walkforward.groupby("strategy", sort=False).size())
    return agg.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="parquet 폴더 (기본: data/processed 또는 data/dummy)")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help="PPO 모델 폴더")
    parser.add_argument("--stage", default="test", choices=STAGES,
                        help="test: 최종 보고용(기본) / val: 선택 단계 모델 평가")
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--lookback", type=int, default=20, help="학습 때와 같아야 함")
    parser.add_argument("--only", nargs="+", default=None, help="일부 변형만 (예: --only simple mdd_l1.0)")
    parser.add_argument("--baselines-only", action="store_true", help="PPO 모델 없이 기준선만 평가")
    parser.add_argument("--out-dir", default=str(MODELS_DIR), help="결과 CSV 저장 폴더")
    args = parser.parse_args()

    data_dir = pick_data_dir(args.data_dir)
    variants = [v for v in PPO_VARIANTS if args.only is None or v["name"] in args.only]
    print(f"데이터 폴더: {data_dir} | stage={args.stage} | windows={args.windows} | seeds={args.seeds}")

    frames, missing = [], []
    for window in args.windows:
        data = load_walk_forward_window(window, data_dir)
        f, m = evaluate_window(data, args.stage, args.lookback, args.seeds, variants,
                               args.models_dir, args.baselines_only)
        frames += f
        missing += m
        print(f"  window {window}: 전략 {len(f)}개 평가 (없는 모델 {len(m)}개)")

    daily = pd.concat(frames)
    daily.index.name = "date"
    keys = ["strategy", "seed", "reward_type", "mdd_lambda"]
    by_window = metrics_table(daily, ["window"] + keys)
    walkforward = metrics_table(daily, keys)                # 윈도우를 이어붙인 전체 기간
    # 모델이 일부 윈도우에만 있으면 기간이 짧아 다른 시드와 비교가 안 된다 -> 요약에서 뺀다
    n_win = by_window.groupby(["strategy", "seed"], sort=False).size().rename("n_windows").reset_index()
    walkforward = walkforward.merge(n_win, on=["strategy", "seed"], how="left")
    walkforward["complete"] = walkforward["n_windows"] == len(args.windows)
    walkforward.insert(0, "windows", " ".join(map(str, args.windows)))
    incomplete = walkforward[~walkforward["complete"]]
    summary = summarize_over_seeds(walkforward[walkforward["complete"]])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_window.to_csv(out_dir / f"eval_{args.stage}_windows.csv", index=False)
    walkforward.to_csv(out_dir / f"eval_{args.stage}_walkforward.csv", index=False)
    summary.to_csv(out_dir / f"eval_{args.stage}_summary.csv", index=False)
    daily.reset_index()[["date", "window", "strategy", "seed", "net_return", "portfolio_value",
                         "drawdown", "turnover", "max_target_weight"]] \
        .to_csv(out_dir / f"eval_{args.stage}_daily.csv", index=False)

    show = ["strategy", "seed", "days", "cum_return", "cagr", "sharpe", "sortino", "max_mdd",
            "calmar", "cvar_95", "avg_turnover", "information_ratio"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(f"\n=== Walk-Forward {args.stage} (windows {args.windows} 연결) ===")
        print(walkforward[show].to_string(index=False))
        print("\n=== 전략별 시드 평균 ===")
        print(summary[["strategy", "n_seeds"] + [f"{m}_mean" for m in
                       ("cum_return", "sharpe", "max_mdd", "calmar", "information_ratio")]]
              .to_string(index=False))

    if len(incomplete) > 0:
        names = ", ".join(f"{r.strategy} s{r.seed}" for r in incomplete.itertuples())
        print(f"\n[경고] 일부 윈도우만 있어 시드 평균에서 제외: {names}")
    if missing:
        print(f"\n[경고] 모델 {len(missing)}개가 없어 건너뜀. 예: {missing[0].name}")
        print(f"  먼저 학습: python train_compare.py --stage {args.stage} "
              f"--windows {' '.join(map(str, args.windows))} --seeds {' '.join(map(str, args.seeds))}")
    print(f"\n결과 저장: {out_dir}/eval_{args.stage}_*.csv")


if __name__ == "__main__":
    main()
