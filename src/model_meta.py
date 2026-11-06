"""PPO 모델 옆에 '어떤 데이터·설정으로 학습했는지'를 JSON으로 남기고, 불러올 때 검사한다.

models/ppo_test_w1_mdd_l1.0_s42.zip  <- SB3 모델
models/ppo_test_w1_mdd_l1.0_s42.json <- 이 파일이 만드는 메타 정보

왜 필요한가:
  - 행동 벡터의 i번째가 어떤 자산인지는 학습 데이터의 자산 순서로 정해진다.
    순서가 다른 데이터(예: 예전 가짜 데이터)로 학습한 모델을 실데이터에 쓰면 비중이 엉뚱한 자산에 들어간다.
  - 관측 차원(RiskTag 포함 여부), lookback 이 다르면 모델이 아예 맞지 않는다.
  - 가짜 데이터로 학습한 모델이 리포트·API에 섞여 들어가는 것을 막는다.
torch를 읽지 않으므로 evaluation-xai·플랫폼에서도 가볍게 쓸 수 있다.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

META_VERSION = 1


def meta_path(model_path):
    """모델 경로(.zip 유무 상관없음) -> 같은 이름의 .json 경로."""
    p = Path(model_path)
    return p.with_suffix(".json") if p.suffix == ".zip" else p.with_name(p.name + ".json")


def build_meta(data, env, *, stage, name, reward_type, mdd_lambda, seed, timesteps, gamma, data_dir):
    """학습 직후 남길 메타 정보 dict."""
    return {
        "meta_version": META_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "window": data.window,
        "variant": name,
        "reward_type": reward_type,
        "mdd_lambda": mdd_lambda,
        "seed": seed,
        "timesteps": timesteps,
        "gamma": gamma,
        "lookback": env.lookback,
        "max_weight": env.max_weight,
        "mdd_limit": env.mdd_limit,
        "tickers": list(data.tickers),                    # 행동/비중 벡터의 자산 순서
        "obs_dim": int(env.observation_space.shape[0]),
        "risk_columns": list(data.risk_columns),          # RiskTag 관측 컬럼 (없으면 [])
        "data_dir": str(data_dir),
        "is_dummy": bool(data.is_dummy),
        "train_start": data.train_start,
        "val_start": data.val_start,
        "train_end": data.train_end,
        "test_end": data.test_end,
    }


def save_meta(model_path, meta):
    with open(meta_path(model_path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def load_meta(model_path):
    """메타 파일을 읽는다. 예전에 학습해 메타가 없는 모델이면 None."""
    path = meta_path(model_path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check_compatible(meta, data, env, allow_dummy=False):
    """모델(meta)이 지금 평가하려는 데이터·환경과 맞는지 검사. 안 맞으면 ValueError."""
    problems = []
    if meta["tickers"] != list(data.tickers):
        problems.append(f"자산 순서가 다름: 모델 {meta['tickers']} / 데이터 {list(data.tickers)}")
    if meta["obs_dim"] != int(env.observation_space.shape[0]):
        problems.append(f"관측 차원이 다름: 모델 {meta['obs_dim']} / 환경 {env.observation_space.shape[0]}")
    if meta["lookback"] != env.lookback:
        problems.append(f"lookback이 다름: 모델 {meta['lookback']} / 환경 {env.lookback}")
    if meta.get("risk_columns", []) != list(data.risk_columns):
        problems.append(f"RiskTag 컬럼이 다름: 모델 {meta.get('risk_columns')} / 데이터 {data.risk_columns}")
    if meta["window"] != data.window:
        problems.append(f"윈도우가 다름: 모델 {meta['window']} / 데이터 {data.window}")
    if meta.get("is_dummy") and not allow_dummy:
        problems.append("가짜(랜덤) 데이터로 학습한 모델입니다")
    if problems:
        raise ValueError("모델과 데이터가 맞지 않습니다:\n  - " + "\n  - ".join(problems))
