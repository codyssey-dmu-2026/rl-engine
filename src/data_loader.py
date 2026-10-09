import json                                # manifest(윈도우 구간 정보) 읽기용
from dataclasses import dataclass          # 결과를 깔끔한 묶음으로 반환하기 위함
from pathlib import Path                   # 경로 처리

import pandas as pd                        # parquet 읽기

# 이 파일(src/data_loader.py) 기준으로 프로젝트 루트의 data/processed 를 기본 경로로 사용
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class WindowData:
    """한 개 walk-forward 윈도우의 학습/테스트 데이터 묶음."""

    window: int                            # 윈도우 번호 (1부터)
    train_features: pd.DataFrame           # 학습 구간 정규화 피처
    train_returns: pd.DataFrame            # 학습 구간 원래 로그수익률 (보상 계산용)
    test_features: pd.DataFrame            # 테스트 구간 정규화 피처
    test_returns: pd.DataFrame             # 테스트 구간 원래 로그수익률
    train_start: str                       # 학습 시작일
    train_end: str                         # 학습 종료일 (exclusive, 이 날부터 테스트)
    test_end: str                          # 테스트 종료일 (exclusive)


def load_walk_forward_window(window: int = 1, data_dir=DEFAULT_DATA_DIR) -> WindowData:
    """features_window{n}.parquet 와 returns.parquet 를 읽어 train/test로 나눈다.

    data-pipeline은 윈도우 파일에 [train_start, test_end) 전체를 담아두므로,
    train_end 날짜를 기준으로 우리가 직접 학습/테스트를 나눠야 한다.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "walk_forward_manifest.json"
    features_path = data_dir / f"features_window{window}.parquet"
    returns_path = data_dir / "returns.parquet"

    # ---------- 필요한 파일이 다 있는지 먼저 확인 ----------
    for path in (manifest_path, features_path, returns_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 파일이 없습니다. data-pipeline의 data/processed/ 결과물을 "
                f"{data_dir} 에 넣었는지 확인하세요."
            )

    # ---------- manifest에서 이 윈도우의 구간 정보 찾기 ----------
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    info = next((w for w in manifest["windows"] if w["window"] == window), None)
    if info is None:
        raise ValueError(f"manifest에 window={window} 가 없습니다")

    # ---------- 데이터 읽기 ----------
    features = pd.read_parquet(features_path)
    all_returns = pd.read_parquet(returns_path)

    # features의 날짜에 맞는 returns만 골라온다 (두 파일의 날짜가 달라지는 사고 방지)
    missing_dates = features.index.difference(all_returns.index)
    if len(missing_dates) > 0:
        raise ValueError(f"returns.parquet에 없는 날짜가 {len(missing_dates)}개 있습니다")
    returns = all_returns.loc[features.index]

    # ---------- train_end 기준으로 학습/테스트 분리 (시간 순서 유지) ----------
    train_end = pd.Timestamp(info["train_end"])
    is_train = features.index < train_end

    return WindowData(
        window=window,
        train_features=features[is_train],
        train_returns=returns[is_train],
        test_features=features[~is_train],
        test_returns=returns[~is_train],
        train_start=info["train_start"],
        train_end=info["train_end"],
        test_end=info["test_end"],
    )
