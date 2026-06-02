"""
6개 탐지기 구현.
검증된 JS 엔진(engine.js)의 알고리즘을 Python/NumPy로 정확히 포팅.
IsolationForest는 scikit-learn 정식 사용. 나머지는 NumPy 벡터화.
"""
from __future__ import annotations
import numpy as np
from .base import BaseDetector, normalize01


# ----------------------------- 통계 유틸 -----------------------------
def _mad(a: np.ndarray, med: float | None = None) -> float:
    med = np.median(a) if med is None else med
    return 1.4826 * np.median(np.abs(a - med))


def estimate_period(series: np.ndarray) -> int:
    """자기상관 1차 피크로 계절성 주기 추정 (수업 check_seasonality 정신)."""
    n = len(series)
    if n < 20:
        return 0
    dev = series - series.mean()
    denom = np.dot(dev, dev) or 1.0
    max_lag = min(n // 3, 200)
    min_lag = max(8, n // 100)
    best_lag, best_acf = 0, 0.0
    for lag in range(min_lag, max_lag):
        acf = np.dot(dev[:-lag], dev[lag:]) / denom
        if acf > best_acf and acf > 0.2:
            best_acf, best_lag = acf, lag
    return best_lag


# ----------------------- 1. Robust Z-Score ------------------------
class RobustZScore(BaseDetector):
    key, name, family = "robustz", "Robust Z-Score", "통계"
    blurb = "중앙값·MAD 기반. 이상치에 오염 안 되는 강건 통계."

    def score(self, series, **kw):
        med = np.median(series)
        mad = _mad(series, med)
        denom = mad if mad > 1e-9 else (series.std() or 1.0)
        return normalize01(np.abs(series - med) / denom)


# ----------------------------- 2. IQR -----------------------------
class IQRFence(BaseDetector):
    key, name, family = "iqr", "IQR Fence", "통계"
    blurb = "사분위 울타리. 비대칭 분포에 강함."

    def score(self, series, **kw):
        q1, q3 = np.percentile(series, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        raw = np.where(series < lo, lo - series,
                       np.where(series > hi, series - hi, 0.0))
        return normalize01(raw)


# ----------------------- 3. Isolation Forest ----------------------
class IsolationForestDetector(BaseDetector):
    key, name, family = "iforest", "Isolation Forest", "ML"
    blurb = "무작위 분할로 고립되는 점을 탐지. 다변량 맥락 (scikit-learn)."

    def score(self, series, window: int = 5, **kw):
        from sklearn.ensemble import IsolationForest
        n = len(series)
        # 슬라이딩 윈도우 특징: [값, 1차차분, 국소표준편차]
        feats = np.zeros((n, 3))
        diff = np.zeros(n); diff[1:] = np.diff(series)
        for i in range(n):
            a, b = max(0, i - window), min(n, i + window + 1)
            feats[i] = [series[i], diff[i], series[a:b].std()]
        # 표준화
        mu, sd = feats.mean(0), feats.std(0)
        sd[sd == 0] = 1
        feats = (feats - mu) / sd
        clf = IsolationForest(n_estimators=100, max_samples=min(256, n),
                              random_state=42, contamination="auto")
        clf.fit(feats)
        # score_samples: 높을수록 정상 → 부호 반전해 이상점수화
        raw = -clf.score_samples(feats)
        return normalize01(raw)


# --------------------- 4. Forecast Residual (AR) ------------------
class ForecastResidual(BaseDetector):
    key, name, family = "forecast", "Forecast Residual", "예측"
    blurb = "AR 예측 잔차 기반 (수업 회귀모형)."

    def __init__(self):
        self._meta = {}

    def score(self, series, lag: int = 8, **kw):
        n = len(series)
        p = min(lag, n // 3)
        if p < 1:
            return np.zeros(n)
        # 설계행렬: [1, y_{t-1}, ..., y_{t-p}]
        rows, y = [], []
        for t in range(p, n):
            rows.append([1.0] + [series[t - k] for k in range(1, p + 1)])
            y.append(series[t])
        X = np.array(rows); y = np.array(y)
        # 정규방정식 + 릿지 정규화 (회귀 PDF의 릿지 개념)
        XtX = X.T @ X + 1e-6 * np.eye(p + 1)
        beta = np.linalg.solve(XtX, X.T @ y)
        self._meta = {"p": int(p)}
        pred = np.full(n, np.nan)
        raw = np.zeros(n)
        for t in range(p, n):
            yh = beta[0] + sum(beta[k] * series[t - k] for k in range(1, p + 1))
            pred[t] = yh
            raw[t] = abs(series[t] - yh)
        # 잔차를 강건표준편차로 스케일
        resid = raw[p:]
        rmad = _mad(resid) or resid.std() or 1.0
        scaled = np.zeros(n)
        scaled[p:] = np.abs(raw[p:]) / rmad
        self._pred = pred
        return normalize01(scaled)

    def meta(self):
        return self._meta


# ----------------------- 5. Level Shift / CUSUM -------------------
class LevelShift(BaseDetector):
    key, name, family = "levelshift", "Level Shift", "변화점"
    blurb = "레벨이 통째로 바뀌는 체제 전환 탐지."

    def score(self, series, window: int | None = None, **kw):
        n = len(series)
        w = window or max(5, n // 20)
        raw = np.zeros(n)
        for i in range(n):
            a = series[max(0, i - w):i]
            b = series[i:min(n, i + w)]
            if len(a) < 2 or len(b) < 2:
                continue
            pooled = np.sqrt((a.var() + b.var()) / 2) or 1.0
            raw[i] = abs(b.mean() - a.mean()) / pooled
        return normalize01(raw)


# ------------------------ 6. Matrix Profile -----------------------
class MatrixProfile(BaseDetector):
    key, name, family = "matrixprofile", "Matrix Profile", "SOTA"
    blurb = "서브시퀀스 최근접거리. 닮은꼴 없는 구간=discord (Yeh 2016)."

    def __init__(self):
        self._meta = {}

    def score(self, series, m: int | None = None, **kw):
        n = len(series)
        if m is None:
            m = estimate_period(series) or max(8, n // 20)
            if m > n / 8:
                m = max(10, m // 2)
        m = max(4, min(m, n // 4))
        L = n - m + 1
        if L < 4:
            self._meta = {"m": int(m)}
            return np.zeros(n)
        # 서브시퀀스 평균·표준편차 (누적합 O(n))
        cum = np.concatenate([[0], np.cumsum(series)])
        cum2 = np.concatenate([[0], np.cumsum(series ** 2)])
        mu = (cum[m:] - cum[:-m]) / m
        var = (cum2[m:] - cum2[:-m]) / m - mu ** 2
        sig = np.sqrt(np.maximum(var, 0))
        sig[sig < 1e-9] = 1e-9
        excl = int(np.ceil(m / 2))
        mp = np.full(L, np.inf)
        # 대각선 점화로 z-정규화 거리^2 (O(n^2))
        for diag in range(1, L):
            qt = np.dot(series[:m], series[diag:diag + m])
            for i in range(L - diag):
                j = i + diag
                if i > 0:
                    qt += series[i + m - 1] * series[j + m - 1] - series[i - 1] * series[j - 1]
                if diag > excl:
                    corr = (qt - m * mu[i] * mu[j]) / (m * sig[i] * sig[j])
                    corr = min(1.0, max(-1.0, corr))
                    dist2 = 2 * m * (1 - corr)
                    if dist2 < mp[i]:
                        mp[i] = dist2
                    if dist2 < mp[j]:
                        mp[j] = dist2
        self._meta = {"m": int(m), "L": int(L)}
        raw = np.zeros(n)
        half = m // 2
        for i in range(L):
            d = np.sqrt(max(0.0, mp[i] if np.isfinite(mp[i]) else 0.0))
            c = i + half
            if c < n:
                raw[c] = d
        # 가장자리 보정
        if half < n:
            raw[:half] = raw[half]
        raw[n - half:] = raw[max(0, n - half - 1)]
        return normalize01(raw)

    def meta(self):
        return self._meta


# ------------------------- 레지스트리 -----------------------------
def build_registry() -> dict[str, BaseDetector]:
    return {d.key: d for d in [
        RobustZScore(), IQRFence(), IsolationForestDetector(),
        ForecastResidual(), LevelShift(), MatrixProfile(),
    ]}
