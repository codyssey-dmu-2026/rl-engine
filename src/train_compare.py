"""보상함수 3종(5개 변형) 비교 실험.

학습 루프는 train.py 의 train_one() 하나만 쓴다 (데이터 로더·시드·저장 경로가 train.py와 항상 같다).

두 단계로 나눠 돌린다:
  1) --stage val  (기본): 각 변형을 fit 구간에서 학습 -> val 구간에서 평가.
                          보상형/lambda 선택은 이 결과로만 한다. (test 성적을 보고 고르지 않기 위함)
  2) --stage test       : 같은 변형을 train 전체(fit+val)로 재학습 -> test 구간에서 평가.
                          리포트·ANOVA에 쓰는 최종 성과.

예)
  python train_compare.py --timesteps 2048                     # 빠른 동작 확인
  python train_compare.py --windows 1 2 3 4 --seeds 42 43 44   # 검증 단계 본 실험
  python train_compare.py --stage test --windows 1 2 3 4 --seeds 42 43 44
"""
import argparse                            # 실행 옵션
import time                                # 실험별 소요 시간 출력

import pandas as pd                        # 결과 표

from data_loader import load_walk_forward_window
from train import MODELS_DIR, STAGES, baseline_metrics, pick_data_dir, train_one

# 명세 요구사항: 보상함수 최소 3종. 변형 3은 lambda 3개로 공격적~방어적 비교
EXPERIMENTS = [
    {"name": "simple",   "reward_type": "simple", "mdd_lambda": 1.0},   # 변형 1: 단순 수익률
    {"name": "sharpe",   "reward_type": "sharpe", "mdd_lambda": 1.0},   # 변형 2: 샤프비율 기반
    {"name": "mdd_l0.5", "reward_type": "mdd",    "mdd_lambda": 0.5},   # 변형 3a: MDD 페널티 약함 (공격적)
    {"name": "mdd_l1.0", "reward_type": "mdd",    "mdd_lambda": 1.0},   # 변형 3b: MDD 페널티 기본값
    {"name": "mdd_l5.0", "reward_type": "mdd",    "mdd_lambda": 5.0},   # 변형 3c: MDD 페널티 강함 (방어적)
]

SHOW_COLS = ["stage", "window", "strategy", "seed", "cum_return", "sharpe", "max_mdd",
             "breach_15pct", "avg_turnover", "max_weight"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None,
                        help="parquet 폴더 (기본: data/processed 실데이터. 가짜는 ../data/dummy 를 직접 지정)")
    parser.add_argument("--risk-file", default=None, help="(선택) RiskTag 관측 parquet")
    parser.add_argument("--stage", default="val", choices=STAGES, help="val: 선택용(기본) / test: 최종 보고용")
    parser.add_argument("--windows", type=int, nargs="+", default=[1], help="walk-forward 윈도우 번호들")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="시드 목록 (ANOVA용으로 여러 개 권장)")
    parser.add_argument("--timesteps", type=int, default=100_000, help="실험당 학습 스텝 (명세: 10만 이상)")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--only", nargs="+", default=None, help="일부 변형만 (예: --only simple mdd_l1.0)")
    args = parser.parse_args()

    experiments = [e for e in EXPERIMENTS if args.only is None or e["name"] in args.only]
    data_dir = pick_data_dir(args.data_dir)
    print(f"데이터 폴더: {data_dir} | stage={args.stage} | windows={args.windows} | seeds={args.seeds}")

    rows = []
    for window in args.windows:
        data = load_walk_forward_window(window, data_dir, risk_file=args.risk_file)
        print(f"\nwindow {window}: fit {len(data.fit_features)}일 / val {len(data.val_features)}일 "
              f"(val 시작 {data.val_start}) / test {len(data.test_features)}일")
        rows.append(baseline_metrics(data, args.lookback, stage=args.stage))     # 동일가중 기준선
        for seed in args.seeds:
            for exp in experiments:
                t0 = time.time()
                m = train_one(data, exp["name"], exp["reward_type"], exp["mdd_lambda"],
                              seed, args.timesteps, args.lookback, stage=args.stage, data_dir=data_dir)
                rows.append(m)
                print(f"  {m['strategy']:>14} s{seed} | 누적 {m['cum_return']:+.2%} | 샤프 {m['sharpe']:+.2f} | "
                      f"MDD {m['max_mdd']:.2%} | 최대비중 {m['max_weight']:.3f} | {time.time() - t0:.0f}s")

    results = pd.DataFrame(rows)
    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"compare_{args.stage}.csv"
    results.to_csv(out, index=False)

    print(f"\n=== {args.stage} 구간 결과 ===")
    print(results[SHOW_COLS].to_string(index=False))

    if args.stage == "val":
        # 선택은 val 결과로만: 변형별 평균 샤프가 가장 높은 것 (윈도우·시드 평균)
        ppo = results[results["strategy"] != "EqualWeight"]
        ranking = ppo.groupby("strategy")["sharpe"].mean().sort_values(ascending=False)
        print("\n[val 기준 변형별 평균 샤프]")
        print(ranking.to_string())
        lam = (ppo[ppo["reward_type"] == "mdd"].groupby("mdd_lambda")["sharpe"].mean())
        if len(lam) > 0:
            print(f"-> val 기준 lambda 선택: {lam.idxmax()}  (test는 이 결정 뒤에 --stage test 로 확인)")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
