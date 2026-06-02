"""
분석 오케스트레이터: 파싱된 시리즈 → 6탐지기 실행 → 임계값 수립 →
합의 → 3지표 평가 → 근거 텍스트 생성까지 묶어 프론트엔드용 JSON을 만든다.
"""
from __future__ import annotations
import numpy as np
from ..detectors.algorithms import build_registry, estimate_period
from . import pipeline as P


def _serialize_scores(arr: np.ndarray, ndigits=4):
    return [round(float(x), ndigits) for x in arr]


def analyze_series(series: np.ndarray, eval_mode: str = "auto") -> dict:
    """한 변수에 대한 전체 탐지·평가."""
    registry = build_registry()
    n = len(series)

    inject = None
    train_mask = test_mask = None
    if eval_mode == "auto":
        contaminated, labels, events = P.inject_anomalies(series, seed=7)
        inject = {"labels": labels, "events": events}
        target = contaminated
        # 시간 순서 train/test 분할 (앞 50% train, 뒤 50% test)
        train_mask, test_mask, _cut = P.make_split(n, train_ratio=0.5)
    else:
        target = series

    detectors_out = {}
    scores = {}
    thresholds = {}
    preds = {}
    holdout = {}
    for key, det in registry.items():
        sc = det.score(target)
        scores[key] = sc
        best = {"th": 0.6}
        roc = None
        if inject is not None:
            # 임계값은 train 구간에서만 선택 (과적합 방지)
            best, roc = P.sweep_threshold(sc, inject["labels"], inject["events"],
                                          train_mask=train_mask)
            # train/test 각각 재평가 → 일반화 갭 측정
            holdout[key] = P.evaluate_holdout(
                sc, inject["labels"], inject["events"],
                best["th"], train_mask, test_mask)
        thresholds[key] = best["th"]
        detectors_out[key] = {
            "score": _serialize_scores(sc),
            "best": best,
            "roc": roc,
            "holdout": holdout.get(key),
            "meta": det.meta(),
            "name": det.name,
            "family": det.family,
        }
        if key == "forecast" and hasattr(det, "_pred"):
            pr = det._pred
            preds["forecast"] = [None if np.isnan(v) else round(float(v), 4) for v in pr]

    # F1 기반 가중치 (평가 모드일 때만, 아니면 균등)
    weights = None
    if inject is not None:
        weights = P.weights_from_f1({k: detectors_out[k]["best"] for k in detectors_out})

    con = P.consensus(scores, thresholds, weights=weights)
    votes = con["votes"]
    n_det = con["n_det"]

    out = {
        "n": n,
        "target": _serialize_scores(target, 4),
        "detectors": detectors_out,
        "thresholds": thresholds,
        "votes": [int(v) for v in votes],
        "soft": _serialize_scores(con["soft"], 4),
        "weighted": _serialize_scores(con["weighted"], 3),
        "weights": {k: round(float(v), 3) for k, v in con["weights_used"].items()},
        "n_detectors": n_det,
        "preds": preds,
    }
    if inject is not None:
        # 합의 N 튜닝 곡선 + 최적 N 추천
        best_n, n_curve = P.sweep_consensus_n(
            votes, inject["labels"], inject["events"], n_det)
        out["consensus_sweep"] = {"best": best_n, "curve": n_curve}
        out["inject"] = {
            "labels": [int(x) for x in inject["labels"]],
            "events": inject["events"],
        }
        # 합의(현재 N=2 기준) train/test 일반화 갭
        out["holdout"] = {
            "detectors": holdout,
            "split": {"train_ratio": 0.5, "n_train": int(train_mask.sum()),
                      "n_test": int(test_mask.sum())},
            "consensus": _consensus_holdout(votes, inject, train_mask, test_mask),
        }
    return out


def _consensus_holdout(votes, inject, train_mask, test_mask, N=2):
    """합의 ≥N표 기준의 train/test F1 갭."""
    import numpy as _np
    flags = (_np.asarray(votes) >= N).astype(_np.int8)
    labels = _np.asarray(inject["labels"])
    ev_tr = P._clip_events(inject["events"], train_mask)
    ev_te = P._clip_events(inject["events"], test_mask)
    m_tr = P.evaluate(labels[train_mask], flags[train_mask], point_adjust=True, events=ev_tr)
    m_te = P.evaluate(labels[test_mask], flags[test_mask], point_adjust=True, events=ev_te)
    return {
        "n": N,
        "train": {"f1": m_tr["f1"], "precision": m_tr["precision"], "recall": m_tr["recall"]},
        "test": {"f1": m_te["f1"], "precision": m_te["precision"], "recall": m_te["recall"]},
        "gap": round(m_tr["f1"] - m_te["f1"], 4),
    }


def analyze_all(parsed: dict, eval_mode: str = "auto") -> dict:
    """모든 변수 분석 + 드리프트 비교 + 근거 텍스트."""
    results = {}
    for col, series in parsed["series"].items():
        results[col] = analyze_series(series, eval_mode)

    active_var = parsed["value_cols"][0]
    sig = P.signature(parsed["series"][active_var])

    return {
        "meta": {
            "n": parsed["n"],
            "time_col": parsed["time_col"],
            "value_cols": parsed["value_cols"],
            "time": parsed["time"],
        },
        "results": results,
        "signature": sig,
        "active_var": active_var,
    }


def generate_explanations(result: dict, time: list, consensus_n: int = 2,
                          time_col=None) -> list:
    """합의 이상 구간에 대한 자연어 근거 자동 생성."""
    votes = np.array(result["votes"])
    target = np.array(result["target"])
    n = len(votes)
    registry = build_registry()
    det_names = {k: d.name for k, d in registry.items()}
    thresholds = result["thresholds"]
    scores = {k: np.array(result["detectors"][k]["score"]) for k in result["detectors"]}

    med = float(np.median(target))
    mad = 1.4826 * np.median(np.abs(target - med)) or target.std() or 1.0

    # 연속 이상 구간 묶기
    events, i = [], 0
    while i < n:
        if votes[i] >= consensus_n:
            j = i
            while j < n and votes[j] >= consensus_n:
                j += 1
            events.append((i, j - 1))
            i = j
        else:
            i += 1

    out = []
    for s, e in events[:12]:
        mid = (s + e) // 2
        max_votes = int(votes[s:e + 1].max())
        agree = [k for k in scores if scores[k][mid] >= thresholds[k]]
        dev = (target[mid] - med) / mad
        length = e - s + 1
        if length <= 1:
            typ = f"순간 스파이크 — 중앙값 대비 {dev:+.1f}σ 편차"
        elif length >= 8:
            typ = f"지속 구간({length} step) — 레벨 이동 또는 드리프트"
        else:
            typ = f"단기 이상 구간({length} step) — 국소 변동성 급증"
        sev = "심각" if max_votes >= 4 else "주의" if max_votes >= 2 else "경미"
        label = time[mid] if (time_col and mid < len(time)) else f"#{mid}"
        out.append({
            "time": label,
            "type": typ,
            "agree": [det_names[k].split()[0] for k in agree],
            "n_agree": len(agree),
            "value": round(float(target[mid]), 2),
            "severity": sev,
            "votes": max_votes,
        })
    return out
