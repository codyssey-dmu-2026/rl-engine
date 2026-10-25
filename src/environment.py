import gymnasium as gym                    # 강화학습 환경 표준 인터페이스
from gymnasium import spaces               # 관측/행동 공간 정의용
import numpy as np                         # 수치 계산
import pandas as pd                        # 데이터프레임 처리

# 지원하는 보상 함수 이름 목록 (오타 방지용)
REWARD_TYPES = ("simple", "sharpe", "mdd")

# 자산별 비중 상한 (팀 설계 문서 backtest.md / decisions.md, evaluation-xai config.max_asset_weight 와 동일)
MAX_ASSET_WEIGHT = 0.4


def softmax(x):
    """임의의 실수 벡터를 합=1, 각 원소>=0 인 비중으로 변환."""
    x = np.clip(np.asarray(x, dtype=np.float64), -10, 10)  # 수치 폭발 방지
    e_x = np.exp(x - np.max(x))                             # 오버플로우 방지
    return e_x / e_x.sum()


def cap_weights(weights, max_weight=MAX_ASSET_WEIGHT):
    """합=1 비중을 '각 자산 <= max_weight' 조건으로 투영한다.

    상한을 넘친 자산은 max_weight로 고정하고, 넘친 만큼을 상한 아래 자산들에
    기존 비중에 비례해 나눠준다. 나눠준 뒤 또 넘치는 자산이 생길 수 있어 반복한다.
    (자산 N개면 최대 N번 반복하면 끝난다)
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    n = len(w)
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"자산 {n}개로는 상한 {max_weight}에서 합=1을 만들 수 없습니다")
    w = w / w.sum()                                         # 혹시 모를 합 오차 보정
    capped = np.zeros(n, dtype=bool)                        # 상한에 고정된 자산 표시
    for _ in range(n):
        over = (w > max_weight) & ~capped                   # 이번에 새로 넘친 자산
        if not over.any():
            break
        capped |= over
        w[capped] = max_weight                              # 넘친 자산은 상한으로 고정
        free = ~capped
        room = 1.0 - max_weight * capped.sum()              # 나머지 자산들이 나눠 가질 비중
        free_sum = w[free].sum()
        if free_sum > 0:
            w[free] *= room / free_sum                      # 기존 비중에 비례해 재분배
        else:
            w[free] = room / free.sum()                     # (이론상 없음) 전부 0이면 균등 분배
    return np.minimum(w, max_weight)                        # 부동소수 오차로 0.4000000001 되는 것 방지


def action_to_weights(action, max_weight=MAX_ASSET_WEIGHT):
    """PPO 행동 -> 목표 비중 (softmax -> 상한 투영).

    학습 환경과 평가(predict_weights)가 반드시 이 함수 하나만 쓰게 해서
    학습 때 비중과 백테스트 때 비중이 어긋나지 않게 한다.
    """
    return cap_weights(softmax(action), max_weight)


class PortfolioEnv(gym.Env):
    """포트폴리오 자산배분 강화학습 환경 (v2: data-pipeline 산출물 전용).

    입력 데이터 2개:
      - features: 정규화된 피처 (features_window{n}.parquet 를 train/test로 자른 것)
                  컬럼: {ticker}_return, {ticker}_rsi, {ticker}_macd, {ticker}_is_market_open
      - returns : 정규화 전 원래 로그수익률 (returns.parquet 중 같은 날짜 구간)
                  컬럼: {ticker}_return  -> 실제 손익 계산(보상)에 사용

    에이전트는 매일 자산별 목표 비중을 정하고, 다음 날 수익률로 보상을 받는다.
    """

    metadata = {"render_modes": []}        # 화면 렌더링은 사용하지 않음

    def __init__(
        self,
        features: pd.DataFrame,            # 정규화된 피처 (행=날짜)
        returns: pd.DataFrame,             # 정규화 전 로그수익률 (행=날짜, features와 같은 인덱스)
        lookback: int = 20,                # 과거 N일 수익률 윈도우 (명세 권장 20~60)
        fee_rate: float = 0.00015,         # 거래 수수료 0.015% (명세 지정값)
        slippage: float = 0.0005,          # 슬리피지 0.05% (명세 지정값)
        reward_type: str = "simple",       # 보상 함수: simple / sharpe / mdd
        mdd_lambda: float = 1.0,           # 변형 3의 MDD 페널티 강도 (권장 0.5~5.0)
        mdd_limit: float = 0.15,           # Safe-Guard: MDD 15% 초과 시 에피소드 조기 종료
        sigma_window: int = 20,            # 샤프 보상에서 변동성을 계산할 최근 일수
        initial_cash: float = 1_000_000.0, # 초기 자본금
        random_start: bool = False,        # True면 에피소드마다 시작 날짜를 무작위로 (학습용)
        min_episode_days: int = 60,        # 무작위 시작 시 최소한 남겨둘 일수
        max_weight: float = MAX_ASSET_WEIGHT,  # 자산별 비중 상한 (0.4)
    ):
        super().__init__()                 # gym.Env 초기화 (필수)

        # ---------- 입력 검증: 잘못된 데이터는 학습 전에 바로 알려준다 ----------
        if reward_type not in REWARD_TYPES:
            raise ValueError(f"reward_type은 {REWARD_TYPES} 중 하나여야 합니다: {reward_type}")
        if not features.index.equals(returns.index):
            raise ValueError("features와 returns의 날짜 인덱스가 다릅니다 (행 개수/날짜 불일치)")
        if returns.shape[1] * max_weight < 1.0:
            raise ValueError(f"자산 {returns.shape[1]}개, 상한 {max_weight}로는 비중 합=1이 불가능합니다")
        if len(features) < lookback + 2:
            raise ValueError(f"데이터가 너무 짧습니다: {len(features)}행 (lookback={lookback})")

        # ---------- 종목 목록: returns 컬럼명에서 티커를 뽑는다 ----------
        # 예: "SPY_return" -> "SPY", "005930_return" -> "005930"
        self.tickers = [c[: -len("_return")] for c in returns.columns]
        self.n_assets = len(self.tickers)                       # 자산 개수 (예: 12)

        # ---------- features를 두 덩어리로 나눈다 ----------
        # (1) 과거 N일을 펼쳐서 볼 정규화 수익률 컬럼
        return_cols = [f"{t}_return" for t in self.tickers]     # returns 순서와 동일하게 맞춤
        missing = [c for c in return_cols if c not in features.columns]
        if missing:
            raise ValueError(f"features에 없는 컬럼: {missing}")
        # (2) 오늘 시점만 볼 나머지 컬럼 (rsi, macd, is_market_open)
        static_cols = [c for c in features.columns if not c.endswith("_return")]

        if features.isna().any().any() or returns.isna().any().any():
            raise ValueError("features/returns에 NaN이 있습니다 (data-pipeline 결과 확인 필요)")

        # ---------- numpy 배열로 변환 (속도 향상) ----------
        self.norm_returns = features[return_cols].to_numpy(dtype=np.float32)    # (T, N) 정규화 수익률
        self.static_features = features[static_cols].to_numpy(dtype=np.float32) # (T, S) 오늘 피처
        # 로그수익률 -> 단순수익률 변환: 비중 가중합은 단순수익률로 해야 정확하다
        self.simple_returns = np.expm1(returns.to_numpy(dtype=np.float64))      # (T, N)
        self.dates = features.index                                             # 날짜 (분석/백테스트용)
        self.n_steps = len(features)                                            # 전체 일수 T
        self.n_static = len(static_cols)                                        # 오늘 피처 개수

        # ---------- 하이퍼파라미터 저장 ----------
        self.lookback = lookback
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.reward_type = reward_type
        self.mdd_lambda = mdd_lambda
        self.mdd_limit = mdd_limit
        self.sigma_window = sigma_window
        self.initial_cash = initial_cash
        self.random_start = random_start
        self.min_episode_days = min_episode_days
        self.max_weight = max_weight

        # ---------- 행동 공간: 자산별 점수 -> softmax로 비중 변환 ----------
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.n_assets,), dtype=np.float32
        )

        # ---------- 관측 공간 ----------
        # [과거 lookback일 수익률 (lookback x N)] + [오늘 rsi/macd/flag] + [현재 비중 N]
        obs_dim = lookback * self.n_assets + self.n_static + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # ---------- 에피소드 상태 변수 (reset에서 초기화) ----------
        self.current_step = 0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = initial_cash
        self.peak_value = initial_cash
        self.return_history = []

    def reset(self, seed=None, options=None):
        """에피소드를 처음 상태로 되돌린다."""
        super().reset(seed=seed)                                # 시드 설정 (재현성)

        first = self.lookback - 1                               # 과거 N일이 모두 채워지는 첫 날
        if self.random_start:
            # 학습용: 에피소드마다 시작일을 무작위로 골라 데이터 전 구간을 골고루 경험하게 한다.
            # (항상 같은 날 시작하면 Safe-Guard 조기종료 때문에 초반 구간만 반복 학습하게 됨)
            last = self.n_steps - 1 - self.min_episode_days     # 이 날 이후에 시작하면 너무 짧음
            self.current_step = int(self.np_random.integers(first, last + 1))
        else:
            self.current_step = first                           # 평가용: 항상 같은 날에서 시작
        self.weights = np.zeros(self.n_assets, dtype=np.float32)  # 처음엔 전부 현금(비중 0)
        self.portfolio_value = self.initial_cash                # 자본금 초기화
        self.peak_value = self.initial_cash                     # 최고점 초기화
        self.return_history = []                                # 수익률 기록 초기화

        return self._get_observation(), {}

    def step(self, action):
        """에이전트의 행동을 받아 하루를 진행한다."""

        # ---------- 1. 행동 -> 목표 비중 (합=1, 0 <= 각 원소 <= 0.4) ----------
        target_weights = action_to_weights(action, self.max_weight)

        # ---------- 2. 거래비용: 비중을 바꾼 만큼 수수료+슬리피지 ----------
        turnover = np.abs(target_weights - self.weights).sum()  # 회전율
        cost = turnover * (self.fee_rate + self.slippage)       # 총 거래비용(비율)

        # ---------- 3. 하루 진행: 다음 날 수익률로 손익 계산 ----------
        self.current_step += 1                                  # 하루 지남
        day_returns = self.simple_returns[self.current_step]    # 그날 자산별 단순수익률
        gross_return = float(np.dot(target_weights, day_returns))  # 비용 전 포트폴리오 수익률
        net_return = gross_return - cost                        # 비용 차감 후 수익률

        # ---------- 4. 자산가치, 낙폭 갱신 ----------
        # current_drawdown: '지금' 고점 대비 얼마나 빠져 있는지 (MDD는 이 값의 기간 중 최댓값)
        self.portfolio_value *= 1.0 + net_return
        self.peak_value = max(self.peak_value, self.portfolio_value)
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # ---------- 5. 비중 드리프트: 하루 지나면 가격 변동으로 비중이 저절로 변한다 ----------
        # 다음 날 거래비용을 정확히 계산하려면 '목표 비중'이 아니라 '변한 비중'이 기준이어야 함
        self.weights = (target_weights * (1.0 + day_returns) / (1.0 + gross_return)).astype(
            np.float32
        )
        self.return_history.append(net_return)

        # ---------- 6. 종료 조건 ----------
        terminated = bool(current_drawdown > self.mdd_limit)    # Safe-Guard: 낙폭 15% 초과
        truncated = bool(self.current_step >= self.n_steps - 1) # 데이터 끝

        # ---------- 7. 보상 ----------
        reward = self._compute_reward(net_return, current_drawdown)
        if terminated:
            reward -= self._safeguard_penalty()                 # 조기 종료가 이득이 되지 않게

        info = {
            "date": self.dates[self.current_step],              # 오늘 날짜
            "portfolio_value": self.portfolio_value,
            "drawdown": current_drawdown,                      # 현재 낙폭 (MDD = 이 값의 최댓값)
            "net_return": net_return,
            "weights": self.weights.copy(),
            "target_weights": target_weights.copy(),            # (float64, 상한 0.4 적용)            # SHAP/리포트용: 에이전트가 정한 비중
            "turnover": turnover,
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        """현재 시점의 관측값: 과거 N일 수익률 + 오늘 지표 + 현재 비중."""
        t = self.current_step
        past = self.norm_returns[t - self.lookback + 1 : t + 1]  # (lookback, N) 오늘 포함 과거 N일
        obs = np.concatenate([past.ravel(), self.static_features[t], self.weights])
        return obs.astype(np.float32)

    def _compute_reward(self, net_return, current_drawdown):
        """보상 함수 3종 (명세 요구사항)."""
        if self.reward_type == "simple":
            return float(net_return)                            # 변형 1: r_t

        if self.reward_type == "sharpe":
            recent = self.return_history[-self.sigma_window :]  # 최근 N일 수익률
            if len(recent) < 5:                                 # 표본이 너무 적으면 변동성이 불안정
                return 0.0
            sigma = float(np.std(recent))                       # 최근 변동성
            return float(net_return / (sigma + 1e-8))           # 변형 2: r_t / sigma_t

        # reward_type == "mdd"
        return float(net_return - self.mdd_lambda * current_drawdown)  # 변형 3: r_t - lambda*DD_t

    def _safeguard_penalty(self):
        """Safe-Guard 조기 종료 시 주는 페널티.

        mdd 보상은 매 스텝 -lambda*DD_t 라서 거의 항상 음수다. 종료 후 보상이 0이면
        '일부러 15%를 깨고 빨리 끝내기'가 누적 보상이 더 높아지는 문제가 생긴다.
        그래서 종료를 '남은 기간 내내 낙폭이 한도(15%)에 머문 것'으로 취급해
        lambda * mdd_limit * (남은 스텝 수) 만큼 한 번에 뺀다.
        끝까지 버티는 정책은 매 스텝 DD <= 15% 이므로 페널티가 이보다 클 수 없다
        -> 조기 종료로 페널티를 피하는 길이 막힌다.

        simple / sharpe 보상은 낙폭을 보상에 넣지 않는 '비교 기준' 변형이라 그대로 둔다.
        """
        if self.reward_type != "mdd":
            return 0.0
        remaining = self.n_steps - 1 - self.current_step        # 데이터 끝까지 남은 스텝
        return float(self.mdd_lambda * self.mdd_limit * remaining)