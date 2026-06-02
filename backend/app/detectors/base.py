"""
탐지기 베이스 — 모든 탐지기는 동일 인터페이스를 따른다 (Strategy 패턴).
    detector.score(series: np.ndarray) -> np.ndarray   # 포인트별 이상점수 [0,1]
이렇게 하면 합의투표·임계값 조정·알고리즘 비교가 일관되게 처리된다.
"""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


def normalize01(arr: np.ndarray) -> np.ndarray:
    """min-max 0~1 정규화. 상수 배열이면 0 반환."""
    arr = np.asarray(arr, dtype=np.float64)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    rng = mx - mn
    if rng < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / rng


class BaseDetector(ABC):
    key: str = "base"
    name: str = "Base"
    family: str = "-"
    blurb: str = ""

    @abstractmethod
    def score(self, series: np.ndarray, **kwargs) -> np.ndarray:
        """포인트별 이상점수를 [0,1]로 정규화해 반환."""
        raise NotImplementedError

    def meta(self) -> dict:
        return {}
