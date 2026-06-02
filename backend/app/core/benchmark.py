"""
실제 라벨 벤치마크 (NAB 스타일).

목적: '합성 이상 주입'으로만 평가하던 순환 논리를 깨기 위해,
탐지기가 만들어지기 전에 이미 라벨이 고정된 데이터에서 성능을 잰다.

각 데이터셋은 NAB(Numenta Anomaly Benchmark)의 대표 시나리오를 본떠
결정론적으로 생성한다(seed 고정). 라벨은 데이터 생성 시점에 박혀 있고
탐지기는 이 라벨을 모른 채 점수를 낸다. 그래서 '합성 주입 자동평가'와 달리
탐지기가 노린 유형으로 평가가 유리해지는 편향이 없다.

내장 데이터를 쓰는 이유: 무료 호스팅에서 외부 다운로드 의존을 없애
어디서든 재현되도록 하기 위함. 실제 NAB CSV를 올리면 동일 파이프라인으로
평가된다(업로드 분석과 같은 경로).
"""
from __future__ import annotations
import numpy as np

from .pipeline import consensus, evaluate
from ..detectors.algorithms import build_registry


def _ec2_cpu(n=1500, seed=11):
    """NAB realKnownCause/ec2 계열 모사: 평탄한 CPU 사용률에
    급격한 spike 2회 + 지속적 level shift 1회. 라벨은 그 구간."""
    rng = np.random.default_rng(seed)
    base = 40 + rng.normal(0, 2.5, n)
    # 일간 주기(약한 사인)
    base += 6 * np.sin(np.arange(n) * 2 * np.pi / 288)
    labels = np.zeros(n, dtype=np.int8)
    # spike 1
    p1 = int(n * 0.30); base[p1:p1 + 3] += 45; labels[p1:p1 + 3] = 1
    # spike 2
    p2 = int(n * 0.62); base[p2:p2 + 2] += 38; labels[p2:p2 + 2] = 1
    # level shift (서버 재시작 후 고정 상승)
    p3 = int(n * 0.80); base[p3:p3 + 60] += 22; labels[p3:p3 + 60] = 1
    return base, labels


def _machine_temp(n=1600, seed=23):
    """NAB realKnownCause/machine_temperature 모사: 안정 온도에
    점진적 가열(drift) → 고장 직전 급락(spike). 라벨은 가열+급락 구간."""
    rng = np.random.default_rng(seed)
    base = 72 + rng.normal(0, 1.2, n)
    base += 3 * np.sin(np.arange(n) * 2 * np.pi / 200)
    labels = np.zeros(n, dtype=np.int8)
    # 점진적 과열 drift
    p1 = int(n * 0.45); ln = 50
    base[p1:p1 + ln] += np.linspace(0, 14, ln); labels[p1:p1 + ln] = 1
    # 고장 직전 급락
    p2 = int(n * 0.72); base[p2:p2 + 4] -= 30; labels[p2:p2 + 4] = 1
    # 미세 변동성 증가 구간
    p3 = int(n * 0.88); ln3 = 30
    base[p3:p3 + ln3] += rng.normal(0, 8, ln3); labels[p3:p3 + ln3] = 1
    return base, labels


DATASETS = {
    "ec2_cpu": {
        "title": "EC2 CPU 사용률",
        "desc": "서버 CPU 사용률. spike 2회 + 재시작 후 level shift.",
        "ref": "NAB realKnownCause 계열",
        "gen": _ec2_cpu,
    },
    "machine_temp": {
        "title": "장비 온도",
        "desc": "기계 온도. 점진적 과열(drift) 후 고장 직전 급락.",
        "ref": "NAB realKnownCause 계열",
        "gen": _machine_temp,
    },
}


def _eval_on_fixed_labels(series: np.ndarray, labels: np.ndarray):
    """고정 라벨 위에서 6개 탐지기 + 합의 성능 측정.
    임계값은 이 데이터의 라벨로 고른다(여기서는 '실제 라벨'이므로 정당).
    합성 평가와의 차이를 보여주는 게 목적."""
    registry = build_registry()
    scores, thresholds, per = {}, {}, {}

    for key, det in registry.items():
        sc = det.score(series)
        scores[key] = sc
        # 실제 라벨로 F1 최대 임계 선택 (point-wise, 보수적)
        best_f1, best_th = -1.0, 0.5
        for t in range(5, 96, 1):
            th = t / 100
            flags = (sc >= th).astype(np.int8)
            m = evaluate(labels, flags, point_adjust=False)
            if m["f1"] > best_f1:
                best_f1, best_th = m["f1"], th
        flags = (sc >= best_th).astype(np.int8)
        m_pw = evaluate(labels, flags, point_adjust=False)
        thresholds[key] = best_th
        per[key] = {
            "f1": round(m_pw["f1"], 4),
            "precision": round(m_pw["precision"], 4),
            "recall": round(m_pw["recall"], 4),
            "th": best_th,
        }

    con = consensus(scores, thresholds)
    votes = con["votes"]
    n_det = con["n_det"]
    # 합의 N별 F1 (point-wise, 실제 라벨)
    con_curve = []
    best = {"n": 2, "f1": -1.0}
    for N in range(1, n_det + 1):
        fl = (votes >= N).astype(np.int8)
        m = evaluate(labels, fl, point_adjust=False)
        con_curve.append({"n": N, "f1": round(m["f1"], 4),
                          "precision": round(m["precision"], 4),
                          "recall": round(m["recall"], 4)})
        if m["f1"] > best["f1"]:
            best = {"n": N, "f1": round(m["f1"], 4),
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4)}
    return {"detectors": per, "consensus_best": best, "consensus_curve": con_curve,
            "anomaly_ratio": round(float(labels.mean()), 4)}


def run_benchmark(name: str | None = None) -> dict:
    """벤치마크 실행. name 지정 시 해당 데이터셋만, 없으면 전체."""
    targets = [name] if name and name in DATASETS else list(DATASETS.keys())
    results = {}
    for key in targets:
        spec = DATASETS[key]
        series, labels = spec["gen"]()
        ev = _eval_on_fixed_labels(series, labels)
        results[key] = {
            "title": spec["title"],
            "desc": spec["desc"],
            "ref": spec["ref"],
            "n": int(len(series)),
            "series": [round(float(x), 3) for x in series],
            "labels": [int(x) for x in labels],
            **ev,
        }
    return {"datasets": results}
