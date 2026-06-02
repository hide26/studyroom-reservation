"""
HAWK 백엔드 — FastAPI
엔드포인트:
  GET  /api/health        상태·탐지기 목록
  POST /api/analyze       CSV 업로드 → 전체 탐지·평가 결과 JSON
  GET  /api/demo          데모 데이터로 분석
  POST /api/explain       합의 N 변경 시 근거 텍스트만 재생성 (경량)
정적 프론트엔드(frontend/)도 같은 서버에서 서빙해 단일 배포 단위 구성.
"""
from __future__ import annotations
import os

# .env 파일에서 환경변수 자동 로드 (있으면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .core.parser import parse_csv, make_demo_csv
from .core.analysis import analyze_all, generate_explanations
from .core.insight import generate_insight
from .detectors.algorithms import build_registry

app = FastAPI(title="HAWK Anomaly API", version="1.0")

# CORS — 프론트가 별도 도메인일 수 있으므로 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_BYTES = 50 * 1024 * 1024  # 50MB

# 분석 결과 캐시 (explain 재요청 시 재분석 방지)
_CACHE: dict = {}


def _darts_available() -> bool:
    try:
        import darts  # noqa
        return True
    except Exception:
        return False


@app.get("/api/health")
def health():
    reg = build_registry()
    key = os.environ.get("OPENAI_API_KEY", "")
    key_ok = bool(key) and key.startswith("sk-") and "여기에" not in key
    return {
        "status": "ok",
        "version": "1.0",
        "detectors": [d.name for d in reg.values()],
        "darts_available": _darts_available(),
        "openai_key_set": key_ok,
    }


def _run(raw: bytes, eval_mode: str, source: str):
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "파일이 너무 큽니다 (최대 50MB).")
    try:
        parsed = parse_csv(raw)
    except Exception as e:
        raise HTTPException(400, f"CSV 파싱 실패: {e}")
    try:
        result = analyze_all(parsed, eval_mode=eval_mode)
    except Exception as e:
        raise HTTPException(500, f"분석 실패: {e}")
    result["source"] = source
    _CACHE["last"] = {"parsed_meta": result["meta"], "results": result["results"]}
    _CACHE["full"] = result
    return result


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), eval_mode: str = Form("auto")):
    raw = await file.read()
    return JSONResponse(_run(raw, eval_mode, file.filename or "uploaded.csv"))


@app.get("/api/demo")
def demo(eval_mode: str = "auto"):
    return JSONResponse(_run(make_demo_csv(), eval_mode, "demo_sensors.csv"))


@app.get("/api/benchmark")
def benchmark(name: str = ""):
    """실제 라벨 벤치마크. 합성 주입과 독립된 고정 정답으로 평가.
    캐시해서 반복 호출 시 재계산 없이 즉시 반환."""
    key = name or "_all"
    cached = _CACHE.get("bench:" + key)
    if cached:
        return JSONResponse(cached)
    from app.core.benchmark import run_benchmark
    result = run_benchmark(name or None)
    _CACHE["bench:" + key] = result
    return JSONResponse(result)


@app.post("/api/explain")
def explain(var: str = Form(...), consensus_n: int = Form(2)):
    """캐시된 결과로 근거 텍스트만 재생성 (재분석 없이 경량)."""
    cache = _CACHE.get("last")
    if not cache or var not in cache["results"]:
        raise HTTPException(404, "먼저 /api/analyze 를 호출하세요.")
    meta = cache["parsed_meta"]
    exps = generate_explanations(
        cache["results"][var], meta["time"], consensus_n, meta["time_col"])
    return {"explanations": exps}


@app.post("/api/insight")
def insight(consensus_n: int = Form(2)):
    """캐시된 분석 결과로 AI 인사이트 생성. 키 없으면 프롬프트 반환(폴백)."""
    full = _CACHE.get("full")
    if not full:
        raise HTTPException(404, "먼저 /api/analyze 또는 /api/demo 를 호출하세요.")
    return generate_insight(full, consensus_n)


# ---- 정적 프론트엔드 서빙 (배포 시 단일 서버) ----
_FRONT = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.isdir(_FRONT):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONT, "index.html"))
    app.mount("/", StaticFiles(directory=_FRONT), name="static")
