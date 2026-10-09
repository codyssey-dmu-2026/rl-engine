from stable_baselines3 import PPO                          # PPO 알고리즘
from stable_baselines3.common.vec_env import DummyVecEnv   # 벡터 환경 래퍼
from stable_baselines3.common.monitor import Monitor       # 에피소드 보상 기록용 래퍼
import pandas as pd                                         # 데이터프레임 처리
import numpy as np                                          # 수치 계산
import os                                                    # 폴더 생성 등 경로 처리

from environment import PortfolioEnv    # 앞서 만든 강화학습 환경


# ================================================================
# 1. 테스트용 가짜 데이터 생성 (나중에 친구 데이터로 교체할 부분)
# ================================================================

n_days = 1000                                               # 총 거래일 수 (학습엔 데이터가 좀 필요함)
n_assets = 5                                                 # 자산 개수
n_features = 25                                              # 피처 개수

dates = pd.date_range("2020-01-01", periods=n_days, freq="B")   # 영업일 기준 날짜 생성

# 랜덤워크로 가짜 주가 생성 (실제 주가처럼 누적곱 형태)
fake_prices = pd.DataFrame(
    100 * np.exp(np.cumsum(np.random.randn(n_days, n_assets) * 0.01, axis=0)),
    index=dates,
    columns=["SPY", "QQQ", "IEF", "GLD", "BIL"],             # 자산명 (친구 데이터 받으면 실제 컬럼명으로 바뀜)
)

# 정규분포 난수로 가짜 피처 생성 (이미 정규화됐다고 가정)
fake_features = pd.DataFrame(
    np.random.randn(n_days, n_features),
    index=dates,
)


# ================================================================
# 2. 실험할 보상 함수 설정 목록 (명세 요구사항: 최소 3종)
# ================================================================

experiments = [
    {"name": "simple",   "reward_type": "simple", "mdd_lambda": 1.0},   # 변형 1: 단순 수익률
    {"name": "sharpe",   "reward_type": "sharpe", "mdd_lambda": 1.0},   # 변형 2: 샤프비율 기반
    {"name": "mdd_l0.5", "reward_type": "mdd",    "mdd_lambda": 0.5},   # 변형 3a: MDD 페널티 약함 (공격적)
    {"name": "mdd_l1.0", "reward_type": "mdd",    "mdd_lambda": 1.0},   # 변형 3b: MDD 페널티 기본값
    {"name": "mdd_l5.0", "reward_type": "mdd",    "mdd_lambda": 5.0},   # 변형 3c: MDD 페널티 강함 (방어적)
]


# ================================================================
# 3. 결과를 저장할 폴더 미리 만들기
# ================================================================

os.makedirs("../models", exist_ok=True)    # models 폴더 없으면 생성 (있으면 에러 없이 넘어감)
os.makedirs("../logs", exist_ok=True)      # logs 폴더도 동일하게 처리


# ================================================================
# 4. 각 실험을 순서대로 학습 및 저장
# ================================================================

results_summary = []    # 나중에 비교표 만들 때 쓸 결과 저장 리스트

for exp in experiments:
    print(f"\n{'='*50}")
    print(f"실험 시작: {exp['name']}  (reward_type={exp['reward_type']}, lambda={exp['mdd_lambda']})")
    print(f"{'='*50}")

    # ---------- 환경 생성 ----------
    env = PortfolioEnv(
        fake_prices,
        fake_features,
        reward_type=exp["reward_type"],      # 이번 실험의 보상 함수 종류
        mdd_lambda=exp["mdd_lambda"],        # 이번 실험의 MDD 페널티 강도
    )
    env = Monitor(env)                       # 에피소드 보상을 자동 기록 (학습곡선용)
    env = DummyVecEnv([lambda: env])         # SB3가 요구하는 벡터 환경 형태로 감싸기

    # ---------- PPO 모델 생성 ----------
    model = PPO(
        policy="MlpPolicy",                  # 기본 다층 퍼셉트론 정책망
        env=env,
        learning_rate=3e-4,                  # 학습률
        n_steps=2048,                        # 업데이트 전 수집할 스텝 수
        batch_size=64,                       # 미니배치 크기
        n_epochs=10,                         # 수집 데이터 재사용 횟수
        gamma=0.99,                          # 할인율
        clip_range=0.2,                      # PPO 핵심 파라미터 (정책 변화 제한폭)
        seed=42,                             # 시드 고정 — 모든 실험에 동일하게 적용해야 공정 비교 가능
        verbose=0,                           # 로그 너무 많이 나오지 않게 끔 (0=끔, 1=켬)
        tensorboard_log=f"../logs/ppo_{exp['name']}",   # 실험별로 로그 폴더 분리
    )

    # ---------- 학습 실행 ----------
    model.learn(total_timesteps=100_000)     # 명세 요구사항: 최소 100,000 스텝

    # ---------- 모델 저장 ----------
    save_path = f"../models/ppo_{exp['name']}"
    model.save(save_path)
    print(f"저장 완료: {save_path}.zip")

    # ---------- 간단 평가: 학습된 모델로 한 번 끝까지 플레이해서 최종 자산가치 확인 ----------
    eval_env = PortfolioEnv(
        fake_prices, fake_features,
        reward_type=exp["reward_type"],
        mdd_lambda=exp["mdd_lambda"],
    )
    obs, info = eval_env.reset(seed=42)      # 평가도 동일 시드로 (공정 비교)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)   # 학습된 정책으로 결정 (탐험 없이)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    # ---------- 결과 기록 ----------
    results_summary.append({
        "실험명": exp["name"],
        "최종_자산가치": info["portfolio_value"],
        "최대_MDD": info["mdd"],
    })

    print(f"최종 자산가치: {info['portfolio_value']:,.0f} | 최대 MDD: {info['mdd']:.2%}")


# ================================================================
# 5. 전체 실험 결과 요약 출력
# ================================================================

print(f"\n{'='*50}")
print("전체 실험 결과 요약")
print(f"{'='*50}")

results_df = pd.DataFrame(results_summary)     # 리스트를 표(DataFrame) 형태로 변환
print(results_df.to_string(index=False))       # 보기 좋게 출력

# 결과를 CSV로도 저장해두면 리포트 작성할 때 바로 활용 가능
results_df.to_csv("../models/experiment_summary.csv", index=False)
print("\n요약 결과 저장 완료: models/experiment_summary.csv")