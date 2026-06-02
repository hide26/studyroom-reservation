"""API 요청/응답 스키마."""
from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, Any


class EvaluateRequest(BaseModel):
    """재평가 요청 — 클라이언트가 합의 N·평가모드만 바꿔 다시 요청할 때."""
    consensus_n: int = 2
    eval_mode: str = "auto"


class HealthResponse(BaseModel):
    status: str
    version: str
    detectors: list[str]
    darts_available: bool
