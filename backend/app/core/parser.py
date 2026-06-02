"""CSV 파싱 + 자동 감지 (pandas 기반)."""
from __future__ import annotations
import io
import numpy as np
import pandas as pd


def parse_csv(raw_bytes: bytes) -> dict:
    """업로드된 CSV 바이트 → 시간축·수치변수 자동 감지된 시리즈 딕셔너리."""
    text = raw_bytes.decode("utf-8-sig", errors="replace")

    # 구분자 자동 감지
    sample = "\n".join(text.splitlines()[:5])
    delim = ","
    best = -1
    for d in [",", "\t", ";", "|"]:
        counts = [len(l.split(d)) for l in sample.splitlines() if l.strip()]
        if counts and all(c == counts[0] for c in counts) and counts[0] > best:
            best, delim = counts[0], d

    df = pd.read_csv(io.StringIO(text), sep=delim)
    if len(df) < 10:
        raise ValueError("데이터가 너무 짧습니다 (최소 10행 필요).")

    # 시간축 감지: 날짜 파싱 성공률 + 컬럼명 힌트
    import warnings
    time_col, time_score = None, 0.5
    for c in df.columns:
        col = df[c].astype(str).head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(col, errors="coerce")
        rate = parsed.notna().mean()
        hint = 0.25 if any(k in str(c).lower()
                           for k in ("time", "date", "stamp", "월", "일", "시간", "index")) else 0
        if rate + hint > time_score:
            time_score, time_col = rate + hint, c

    # 수치형 변수 감지
    value_cols = []
    for c in df.columns:
        if c == time_col:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        if num.notna().mean() > 0.8:
            value_cols.append(c)
    if not value_cols:
        raise ValueError("수치형 변수를 찾지 못했습니다.")

    # 시간축
    if time_col is not None:
        time = df[time_col].astype(str).tolist()
    else:
        time = [str(i) for i in range(len(df))]

    # 각 변수 → 결측 선형보간
    series = {}
    for c in value_cols:
        arr = pd.to_numeric(df[c], errors="coerce").interpolate(
            limit_direction="both").to_numpy(dtype=np.float64)
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)) if not np.isnan(arr).all() else 0.0)
        series[str(c)] = arr

    return {
        "time": time,
        "series": series,
        "n": len(df),
        "time_col": str(time_col) if time_col is not None else None,
        "value_cols": [str(c) for c in value_cols],
    }


def make_demo_csv() -> bytes:
    """데모용 다변량 센서 CSV 생성 (자연 이상 포함)."""
    n = 480
    rng = np.random.default_rng(0)
    t0 = pd.Timestamp("2026-01-01")
    rows = []
    for i in range(n):
        ts = (t0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M")
        temp = 22 + 5 * np.sin(2 * np.pi * i / 24) + 0.01 * i + (rng.random() - .5) * 1.2
        pres = 101 + 2 * np.sin(2 * np.pi * i / 24 + 1) + (rng.random() - .5) * 0.6
        vib = 0.5 + 0.3 * np.sin(2 * np.pi * i / 12) + (rng.random() - .5) * 0.15
        flow = 80 + 10 * np.sin(2 * np.pi * i / 48) + (rng.random() - .5) * 4
        if i == 130: temp += 14
        if 200 <= i < 218: pres += 6
        if i == 310: vib += 2.2
        if 360 <= i < 385: flow += 0.8 * (i - 360)
        rows.append(f"{ts},{temp:.3f},{pres:.3f},{vib:.4f},{flow:.2f}")
    header = "timestamp,temp_sensor,pressure,vibration,flow_rate"
    return (header + "\n" + "\n".join(rows)).encode("utf-8")
