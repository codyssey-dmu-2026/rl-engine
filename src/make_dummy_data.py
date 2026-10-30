"""data-pipeline 산출물과 '같은 형식'의 가짜(랜덤) 데이터를 만든다.

예종님 실제 데이터를 받기 전에 환경/학습 코드를 먼저 검증하기 위한 용도다.
저장 위치: <프로젝트 루트>/data/dummy/   (실제 데이터 data/processed/ 와 섞이지 않게 분리)
"""
import json                                # manifest 저장용
from pathlib import Path                   # 경로 처리

import numpy as np                         # 난수 생성
import pandas as pd                        # 데이터프레임

# 예종님 config.py 의 자산 유니버스(12개)와 워크포워드 윈도우를 그대로 따라한다.
# 자산 순서는 실데이터 returns.parquet 컬럼 순서(문자열 정렬)와 같게 맞춘다
# -> 순서가 다르면 행동 벡터의 i번째가 다른 자산을 가리켜 모델을 섞어 쓸 수 없다.
TICKERS = sorted(["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TLT", "GLD", "VNQ",
                  "005930", "000660", "069500", "148070"])
WINDOWS = [
    ("2018-01-01", "2022-01-01", "2023-01-01"),
    ("2018-01-01", "2023-01-01", "2024-01-01"),
    ("2018-01-01", "2024-01-01", "2025-01-01"),
    ("2018-01-01", "2025-01-01", "2026-01-01"),
]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "dummy"


def main(seed: int = 0, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)                        # 재현 가능한 난수
    dates = pd.bdate_range("2018-01-02", "2025-12-31")       # 영업일 기준 날짜
    n = len(dates)

    # ---------- 정규화 전 원본 피처 만들기 ----------
    raw = {}
    for i, t in enumerate(TICKERS):
        drift = 0.0002 + 0.0001 * (i % 3)                    # 자산마다 약간 다른 평균 수익률
        vol = 0.008 + 0.002 * (i % 4)                        # 자산마다 다른 변동성
        ret = rng.normal(drift, vol, n)                      # 일간 로그수익률
        raw[f"{t}_return"] = ret
        raw[f"{t}_rsi"] = np.clip(50 + np.cumsum(rng.normal(0, 3, n)) % 40 + 5, 0, 100)
        raw[f"{t}_macd"] = rng.normal(0, 0.005, n)
    raw_df = pd.DataFrame(raw, index=dates)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- returns.parquet: 정규화 전 로그수익률만 ----------
    return_cols = [c for c in raw_df.columns if c.endswith("_return")]
    raw_df[return_cols].to_parquet(out_dir / "returns.parquet")

    # ---------- 윈도우별 features: 학습 구간 통계로만 z-score ----------
    flags = pd.DataFrame(1, index=dates, columns=[f"{t}_is_market_open" for t in TICKERS])
    manifest = {"n_std": 5.0, "is_dummy": True, "windows": []}   # is_dummy: 로더가 경고를 띄우는 표시
    for i, (train_start, train_end, test_end) in enumerate(WINDOWS, start=1):
        train = raw_df[(raw_df.index >= train_start) & (raw_df.index < train_end)]
        window_slice = raw_df[(raw_df.index >= train_start) & (raw_df.index < test_end)]
        normalized = (window_slice - train.mean()) / train.std()      # 학습 구간 통계만 사용
        normalized = pd.concat([normalized, flags.loc[normalized.index]], axis=1)
        normalized.to_parquet(out_dir / f"features_window{i}.parquet")
        manifest["windows"].append({
            "window": i,
            "path": f"features_window{i}.parquet",
            "train_start": train_start,
            "train_end": train_end,
            "test_end": test_end,
        })

    with open(out_dir / "walk_forward_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"가짜 데이터 생성 완료 -> {out_dir}")
    print(f"  returns.parquet: {len(return_cols)}개 종목, {n}일")
    print(f"  features_window1~{len(WINDOWS)}.parquet + walk_forward_manifest.json")


if __name__ == "__main__":
    main()
