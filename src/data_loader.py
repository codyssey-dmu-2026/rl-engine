import json                                # manifest(윈도우 구간 정보) 읽기용
import warnings                            # 가짜 데이터 사용 경고
from dataclasses import dataclass, field   # 결과를 깔끔한 묶음으로 반환하기 위함
from pathlib import Path                   # 경로 처리

import pandas as pd                        # parquet 읽기

# 이 파일(src/data_loader.py) 기준으로 프로젝트 루트의 data/processed 를 기본 경로로 사용
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT / "data" / "processed"     # data-pipeline 실데이터
DUMMY_DATA_DIR = ROOT / "data" / "dummy"           # make_dummy_data.py 가짜 데이터 (테스트 전용)
RISK_SUFFIX = "_risk"                              # RiskTag 관측 컬럼 이름: {ticker}_risk

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
    tickers: list = field(default_factory=list)  # 자산 순서 (returns 컬럼 순서 = 행동/비중 순서)
    is_dummy: bool = False                 # make_dummy_data.py 로 만든 가짜 데이터인지
    risk_columns: list = field(default_factory=list)  # 관측에 붙인 RiskTag 컬럼 (없으면 빈 리스트)


def resolve_data_dir(cli_value=None):
    """사용할 데이터 폴더를 정한다. 실데이터가 없으면 '조용히' 가짜로 바꾸지 않고 멈춘다.

    - --data-dir 를 주면 그 폴더를 쓴다 (가짜 데이터는 이렇게 명시해야만 쓸 수 있다).
    - 안 주면 data/processed (실데이터). 없으면 에러.
    """
    if cli_value:
        return Path(cli_value)
    if (DEFAULT_DATA_DIR / "walk_forward_manifest.json").exists():
        return DEFAULT_DATA_DIR
    raise FileNotFoundError(
        f"실데이터가 없습니다: {DEFAULT_DATA_DIR}\n"
        "  - data-pipeline의 data/processed/ 결과물(팀 공용 스냅샷)을 이 폴더에 넣거나\n"
        f"  - 동작 확인만 하려면 가짜 데이터를 명시하세요: --data-dir {DUMMY_DATA_DIR}"
    )


def load_risk_features(risk_file, index, tickers):
    """RiskTag 관측 파일을 읽어 features와 같은 날짜·자산 순서로 맞춘다.

    파일 형식(parquet): index=날짜, 컬럼={ticker}_risk (자산마다 1개, 값이 클수록 위험).
    각 행의 값은 '그 날짜의 결정 시각에 이미 알 수 있던 태그'만으로 만들어야 한다
    (결정 시각 규칙은 agentic-research와 합의한 것을 따른다 -> 미래 정보 누수 방지).
    날짜가 없는 행은 0(태그 없음)으로 채우고 몇 개인지 경고한다.
    """
    risk = pd.read_parquet(risk_file)
    cols = [f"{t}{RISK_SUFFIX}" for t in tickers]
    missing = [c for c in cols if c not in risk.columns]
    if missing:
        raise ValueError(f"{risk_file}에 없는 RiskTag 컬럼: {missing}")
    aligned = risk[cols].reindex(index)
    n_missing = int(aligned.isna().any(axis=1).sum())
    if n_missing:
        warnings.warn(f"RiskTag가 없는 날짜 {n_missing}개를 0으로 채움 ({risk_file})")
    return aligned.fillna(0.0).astype("float32")


def load_walk_forward_window(
    window: int = 1, data_dir=DEFAULT_DATA_DIR, val_months: int = DEFAULT_VAL_MONTHS,
    risk_file=None,
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
    tickers = [c[: -len("_return")] for c in returns.columns]

    # ---------- (선택) RiskTag 관측: features 오른쪽에 {ticker}_risk 컬럼으로 붙인다 ----------
    # 환경은 '_return'이 아닌 컬럼을 모두 오늘 피처로 관측에 넣으므로 288 -> 300차원이 된다.
    risk_columns = []
    if risk_file is not None:
        risk = load_risk_features(risk_file, features.index, tickers)
        features = pd.concat([features, risk], axis=1)
        risk_columns = list(risk.columns)

    # 예전에 만든 가짜 데이터는 manifest에 표시가 없어서 폴더 위치로도 판단한다
    is_dummy = bool(manifest.get("is_dummy", False)) or data_dir.resolve() == DUMMY_DATA_DIR.resolve()
    if is_dummy:
        warnings.warn(f"가짜(랜덤) 데이터로 실행 중입니다: {data_dir} -> 결과를 리포트에 쓰면 안 됩니다")

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
        tickers=tickers,
        is_dummy=is_dummy,
        risk_columns=risk_columns,
    )
