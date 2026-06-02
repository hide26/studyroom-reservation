"""
AI 인사이트 생성 — HAWK 탐지 결과를 '데이터를 어느 정도 읽을 줄 아는 중급자'가
이해하기 쉬운 해석으로 변환한다.

설계 원칙:
  - AI는 '해석'만 한다. 탐지·점수·F1 계산은 전부 HAWK가 이미 수행했고,
    AI는 그 수치를 읽어 설명·우선순위·다음 단계만 제시한다.
  - 환각 차단: 제공된 수치 외의 값은 만들지 않도록 시스템 프롬프트에서 강제.
  - 과용 방지: 인사이트 버튼을 눌렀을 때만 1회 호출.
"""
from __future__ import annotations
import os
import json
import numpy as np
from .pipeline import evaluate, evaluate_all, consensus
from ..detectors.algorithms import build_registry

DET_NAMES = {d.key: d.name for d in build_registry().values()}


# ---------------------------------------------------------------------------
# 1) HAWK 결과(JSON) → AI에게 넘길 구조화된 요약 dict
#    여기서 '우리가 실제로 계산한 수치'만 추출한다. AI는 이 안의 숫자만 인용한다.
# ---------------------------------------------------------------------------
def build_context(analysis: dict, consensus_n: int = 2) -> dict:
    meta = analysis["meta"]
    results = analysis["results"]
    time = meta["time"]
    n = meta["n"]
    time_range = f"{time[0]} ~ {time[-1]}" if meta["time_col"] else f"인덱스 0 ~ {n-1}"

    active_var = analysis["active_var"]

    # --- 변수별 진단 ---
    per_variable = []
    total_anom = 0
    for col, R in results.items():
        votes = np.array(R["votes"])
        n_anom = int(np.sum(votes >= consensus_n))
        total_anom = total_anom if col != active_var else total_anom  # noqa
        # 반응한 탐지기: 임계값 이상 발화가 있었던 탐지기
        reacting = []
        for k, det in R["detectors"].items():
            sc = np.array(det["score"])
            th = R["thresholds"][k]
            fired = int(np.sum(sc >= th))
            if fired > 0:
                reacting.append({"detector": DET_NAMES[k], "fired": fired,
                                 "f1": round(det["best"].get("f1", 0), 3) if det.get("best") else None})
        reacting.sort(key=lambda x: -(x["f1"] or 0))
        # 가장 강한 이상 시점
        peak_idx = int(np.argmax(votes)) if len(votes) else 0
        per_variable.append({
            "variable": col,
            "anomaly_count": n_anom,
            "anomaly_ratio": round(n_anom / n * 100, 1),
            "reacting_detectors": reacting[:4],
            "max_votes": int(votes.max()) if len(votes) else 0,
            "peak_time": time[peak_idx] if meta["time_col"] and peak_idx < len(time) else f"#{peak_idx}",
        })
    per_variable.sort(key=lambda x: -x["anomaly_ratio"])

    # --- 활성 변수 기준 평가지표 + 합성주입 유형별 재현율 ---
    Rv = results[active_var]
    metrics = None
    injection_recall = None
    if Rv.get("inject"):
        labels = np.array(Rv["inject"]["labels"])
        events = Rv["inject"]["events"]
        votes = np.array(Rv["votes"])
        flags = (votes >= consensus_n).astype(np.int8)
        ev = evaluate_all(labels, flags, events)
        metrics = {
            "pointwise_f1": round(ev["pointwise"]["f1"], 3),
            "pointwise_precision": round(ev["pointwise"]["precision"], 3),
            "pointwise_recall": round(ev["pointwise"]["recall"], 3),
            "adjusted_f1": round(ev["adjusted"]["f1"], 3),
            "affiliation_f1": round(ev["affiliation"]["f1"], 3),
            "f1_gap": round(ev["adjusted"]["f1"] - ev["pointwise"]["f1"], 3),
        }
        # 유형별 탐지율 (주입 이벤트 중 하나라도 탐지된 비율)
        by_type = {}
        for e in events:
            seg = flags[e["pos"]:min(n, e["pos"] + e["len"])]
            hit = int(seg.any())
            t = e["type"]
            by_type.setdefault(t, [0, 0])
            by_type[t][0] += hit
            by_type[t][1] += 1
        type_kr = {"spike": "Spike(스파이크)", "levelshift": "Level Shift(레벨 이동)",
                   "variance": "Variance Burst(분산 폭발)", "drift": "Drift(드리프트)"}
        injection_recall = {type_kr.get(t, t): f"{c[0]}/{c[1]} ({round(c[0]/c[1]*100)}%)"
                            for t, c in by_type.items()}

    # --- 합의 분포 ---
    votes_v = np.array(Rv["votes"])
    vote_hist = {int(v): int(np.sum(votes_v == v)) for v in range(Rv["n_detectors"] + 1)}

    return {
        "filename": analysis.get("source", "uploaded.csv"),
        "n_points": n,
        "variables": meta["value_cols"],
        "time_range": time_range,
        "consensus_threshold": consensus_n,
        "n_detectors": Rv["n_detectors"],
        "active_variable": active_var,
        "total_anomalies_active": int(np.sum(votes_v >= consensus_n)),
        "anomaly_ratio_active": round(float(np.sum(votes_v >= consensus_n)) / n * 100, 1),
        "vote_distribution": vote_hist,
        "per_variable": per_variable,
        "metrics": metrics,
        "injection_recall": injection_recall,
    }


# ---------------------------------------------------------------------------
# 2) 시스템 프롬프트 — 역할·독자·규칙을 고정 (AI를 '해석자'로 좁게 가둠)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """너는 시계열 이상탐지 결과를 해석하는 데이터 분석가다.

[독자] 데이터를 어느 정도 읽을 줄 아는 '중급자'. 따라서:
- F1, Precision/Recall, 표준편차(σ) 같은 기본 통계 용어는 설명 없이 그대로 써도 된다.
- 단, Matrix Profile, point-adjusted F1, affiliation F1처럼 생소한 개념은 한 줄로 짧게 풀어준다.
- 숫자만 나열하지 말고 항상 "그래서 이게 무슨 뜻인지"를 함께 말한다.
- 너무 학술적이지도, 너무 떠먹여주지도 않는 중간 톤을 유지한다.

[엄격한 규칙]
1. 제공된 수치 외의 값을 절대 만들지 마라. 모르면 "데이터에 없음"이라고 솔직히 써라.
2. 탐지·계산은 이미 HAWK 시스템이 수행했다. 너는 그 결과를 '해석'만 한다. 새로 탐지하거나 점수를 추정하지 마라.
3. 모든 주장에 제공된 수치를 근거로 인용하라 (예: "temp_sensor 이상률 12.3%").
4. point-wise F1과 point-adjusted F1 차이가 0.15 이상 크면, point-adjusted가 과대추정 경향이 있다는 점을 반드시 경고하라 (Kim et al. 2022).
5. 한국어로 작성. 과장·홍보성 표현 금지. 불확실성은 솔직하게.

[출력 형식] 아래 5개 섹션을 마크다운 헤더(##)로 구분해 작성:
## 1. 핵심 요약
2~3문장. 이 데이터에서 어떤 이상이 발견됐는지 한마디로.
## 2. 변수별 진단
어떤 변수가 어떤 유형의 이상을 얼마나 보이는지, 심각도와 함께.
## 3. 탐지기 합의 해석
6개 중 몇 개가 동의했는지. 많이 동의할수록 신뢰도가 높다는 맥락으로.
## 4. 평가지표 해석
세 F1의 의미와 차이. 격차가 크면 위 규칙 4를 적용.
## 5. 다음 단계 제안
중급 분석자가 추가로 무엇을 살펴봐야 하는지 (현장 지시가 아니라 분석적 다음 단계)."""


def build_user_prompt(ctx: dict) -> str:
    """구조화 컨텍스트를 사람이 읽는 형태로 직렬화해 유저 프롬프트로."""
    lines = []
    lines.append("아래는 HAWK가 분석한 결과다. 이 수치만 사용해 인사이트를 작성하라.\n")
    lines.append("[데이터 정보]")
    lines.append(f"- 파일명: {ctx['filename']}")
    lines.append(f"- 데이터 포인트: {ctx['n_points']}개")
    lines.append(f"- 분석 변수: {', '.join(ctx['variables'])}")
    lines.append(f"- 시간 범위: {ctx['time_range']}")
    lines.append("")
    lines.append("[탐지 결과 — 기준 변수: %s]" % ctx["active_variable"])
    lines.append(f"- 합의 임계값: {ctx['consensus_threshold']}표 (전체 {ctx['n_detectors']}개 탐지기 중)")
    lines.append(f"- 탐지된 이상 시점: {ctx['total_anomalies_active']}개 (전체의 {ctx['anomaly_ratio_active']}%)")
    lines.append(f"- 합의 분포(동의 탐지기 수: 포인트 수): {ctx['vote_distribution']}")
    lines.append("")
    lines.append("[변수별 상세]")
    for v in ctx["per_variable"]:
        dets = ", ".join(f"{d['detector']}(F1 {d['f1']})" for d in v["reacting_detectors"]) or "없음"
        lines.append(f"- {v['variable']}: 이상 {v['anomaly_count']}개({v['anomaly_ratio']}%), "
                     f"최대 합의 {v['max_votes']}표, 최강 시점 {v['peak_time']}, 반응 탐지기: {dets}")
    lines.append("")
    if ctx["metrics"]:
        m = ctx["metrics"]
        lines.append("[평가지표 — 합성 이상 주입 기준]")
        lines.append(f"- Point-wise F1: {m['pointwise_f1']} (P {m['pointwise_precision']} / R {m['pointwise_recall']})")
        lines.append(f"- Point-adjusted F1: {m['adjusted_f1']}")
        lines.append(f"- Affiliation F1: {m['affiliation_f1']}")
        lines.append(f"- PA−PW F1 격차: {m['f1_gap']}")
        lines.append("")
    if ctx["injection_recall"]:
        lines.append("[합성 주입 유형별 탐지율]")
        for t, r in ctx["injection_recall"].items():
            lines.append(f"- {t}: {r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3) OpenAI 호출
# ---------------------------------------------------------------------------
def generate_insight(analysis: dict, consensus_n: int = 2,
                     model: str = "gpt-4o-mini") -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    ctx = build_context(analysis, consensus_n)
    user_prompt = build_user_prompt(ctx)

    # 키 미설정 / placeholder 값이면 프롬프트 복사 폴백
    key_ok = bool(api_key) and api_key.startswith("sk-") and "여기에" not in api_key
    if not key_ok:
        return {
            "ok": False,
            "reason": "no_api_key",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "context": ctx,
        }

    # 모델 폴백 순서: 요청 모델 → gpt-4o-mini → gpt-4o
    candidates = [model]
    for fb in ("gpt-4o-mini", "gpt-4o"):
        if fb not in candidates:
            candidates.append(fb)

    last_err = None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"ok": False, "reason": "api_error",
                "error": _humanize_error(e),
                "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt, "context": ctx}

    for m in candidates:
        try:
            resp = client.chat.completions.create(
                model=m,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,   # 해석 일관성 위해 낮게
                max_tokens=1400,
            )
            text = resp.choices[0].message.content
            return {"ok": True, "insight": text, "model": m, "context": ctx}
        except Exception as e:
            last_err = e
            # 모델 없음 류 오류만 다음 후보로, 그 외(키/잔액)는 즉시 중단
            msg = str(e).lower()
            if "model" in msg and ("not found" in msg or "does not exist" in msg):
                continue
            break

    return {"ok": False, "reason": "api_error",
            "error": _humanize_error(last_err),
            "system_prompt": SYSTEM_PROMPT, "user_prompt": user_prompt, "context": ctx}


def _humanize_error(e) -> str:
    """OpenAI 예외를 사용자가 알아볼 한국어 진단으로 변환."""
    s = str(e)
    low = s.lower()
    if "insufficient_quota" in low or "exceeded your current quota" in low:
        return ("OpenAI 크레딧/쿼터가 부족합니다. platform.openai.com → Billing 에서 "
                "결제수단·잔액을 확인하세요. (크레딧 구매 후 반영까지 시간이 걸릴 수 있습니다.)")
    if "invalid_api_key" in low or "incorrect api key" in low or ("401" in s and "key" in low):
        return "API 키가 올바르지 않습니다. .env 의 OPENAI_API_KEY 값을 다시 확인하세요."
    if "rate limit" in low or "429" in s:
        return "요청이 일시적으로 제한되었습니다(rate limit). 잠시 후 다시 시도하세요."
    if "model" in low and ("not found" in low or "does not exist" in low):
        return "요청한 모델을 사용할 수 없습니다. 계정에 모델 접근 권한이 있는지 확인하세요."
    if "connection" in low or "host" in low or "timeout" in low:
        return f"OpenAI 서버에 연결하지 못했습니다(네트워크). 원본 오류: {s}"
    return f"OpenAI 호출 오류: {s}"
