import json                                # manifest(윈도우 구간 정보) 읽기용
from dataclasses import dataclass          # 결과를 깔끔한 묶음으로 반환하기 위함
from pathlib import Path                   # 경로 처리

import pandas as pd                        # parquet 읽기

# 이 파일(src/data_loader.py) 기준으로 프로젝트 루트의 data/processed 를 기본 경로로 사용
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# 학습 구간 끝에서 검증용으로 떼어낼 기간 (validation.md 게이트 3)
DEFAULT_VAL_MONTHS = 6


@dataclass
class WindowData:
    """한 개 walk-forward 윈도우의 학습/검증/테스트 데이터 묶음.

    날짜 순서:  [train_start ......... val_start ...... train_end) [train_end ... test_end)
                 |<------- fit ------->|<---- val ---->|           |<------ test ------>|
                 |<-------------- train (= fit + val) ------------>|

    - fit  : 보상형/lambda/하이퍼파라미터 '후보'를 학습하는 구간
    - val  : 후보 중 무엇을 고를지 정하는 구간 (선택은 여기서만!)
    - train: 고른 설정으로 test 직전까지 전체를 다시 학습하는 구간
    - test : 최종 성과 보고용. 선택에는 절대 쓰지 않는다.
    """

    window: int                            # 윈도우 번호 (1부터)
    train_features: pd.DataFrame           # 학습 구간 전체(fit+val) 정규화 피처 -> 최종 재학습용
    train_returns: pd.DataFrame            # 학습 구간 전체 원래 로그수익률 (보상 계산용)
    fit_features: pd.DataFrame             # 학습 구간 중 검증 이전 부분 -> 후보 학습용
    fit_returns: pd.DataFrame
    val_features: pd.DataFrame             # 학습 구간 마지막 val_months -> 선택용
    val_returns: pd.DataFrame
    test_features: pd.DataFrame            # 테스트 구간 정규화 피처
    test_returns: pd.DataFrame             # 테스트 구간 원래 로그수익률
    train_start: str                       # 학습 시작일
    val_start: str                         # 검증 시작일 (이 날부터 val)
    train_end: str                         # 학습 종료일 (exclusive, 이 날부터 테스트)
    test_end: str                          # 테스트 종료일 (exclusive)


def load_walk_forward_window(
    window: int = 1, data_dir=DEFAULT_DATA_DIR, val_months: int = DEFAULT_VAL_MONTHS
) -> WindowData:
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

    # ---------- 학습 구간 마지막 val_months 개월을 검증용으로 분리 ----------
    val_start = train_end - pd.DateOffset(months=val_months)
    is_fit = features.index < val_start
    is_val = is_train & ~is_fit
    if is_fit.sum() == 0 or is_val.sum() == 0:
        raise ValueError(f"window={window}: 학습 구간이 짧아 검증 {val_months}개월을 뗄 수 없습니다")

    return WindowData(
        window=window,
        train_features=features[is_train],
        train_returns=returns[is_train],
        fit_features=features[is_fit],
        fit_returns=returns[is_fit],
        val_features=features[is_val],
        val_returns=returns[is_val],
        test_features=features[~is_train],
        test_returns=returns[~is_train],
        train_start=info["train_start"],
        val_start=str(val_start.date()),
        train_end=info["train_end"],
        test_end=info["test_end"],
    )
