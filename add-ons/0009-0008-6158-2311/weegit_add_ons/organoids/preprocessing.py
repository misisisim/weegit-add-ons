from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import numpy as np
from pydantic import BaseModel, Field

from weegit.core.conversions.filters import (
    ButterworthBandPassFilter,
    ButterworthHighPassFilter,
    ButterworthLowPassFilter,
    NotchFilter,
)

from .common import DEFAULT_PIPELINE_NAME, PIPELINES_FILENAME


class PreprocessingStep(BaseModel):
    """One ordered preprocessing stage for a channel matrix."""

    kind: str
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineSpec(BaseModel):
    name: str
    steps: List[PreprocessingStep] = Field(default_factory=list)
    description: str = ""


def default_pipeline_store() -> Dict[str, PipelineSpec]:
    return {DEFAULT_PIPELINE_NAME: PipelineSpec(name=DEFAULT_PIPELINE_NAME, steps=[], description="No preprocessing")}


def pipelines_file(add_on_data_dir: Path) -> Path:
    return Path(add_on_data_dir).parent / "preprocessing_pipeline" / PIPELINES_FILENAME


def read_pipeline_store(path_or_dir: Path) -> Dict[str, PipelineSpec]:
    path = Path(path_or_dir)
    if path.is_dir() or path.name != PIPELINES_FILENAME:
        path = pipelines_file(path)
    if not path.exists():
        return default_pipeline_store()
    raw = json.loads(path.read_text(encoding="utf-8"))
    store: Dict[str, PipelineSpec] = {}
    for name, payload in (raw or {}).items():
        try:
            spec = PipelineSpec.model_validate(payload)
        except Exception:
            continue
        clean_name = spec.name.strip() or str(name)
        spec.name = clean_name
        store[clean_name] = spec
    if DEFAULT_PIPELINE_NAME not in store:
        store.update(default_pipeline_store())
    return store


def write_pipeline_store(path_or_dir: Path, pipelines: Mapping[str, PipelineSpec]) -> Path:
    path = Path(path_or_dir)
    if path.is_dir() or path.name != PIPELINES_FILENAME:
        path = pipelines_file(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: spec.model_dump() for name, spec in pipelines.items()}
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def _filter_from_step(step: PreprocessingStep):
    params = dict(step.params or {})
    kind = step.kind.strip().lower()
    if kind == "notch":
        flt = NotchFilter()
        flt.notch_freq_hz = float(params.get("notch_freq_hz", params.get("frequency_hz", 50.0)))
        flt.q_factor = float(params.get("q_factor", 30.0))
    elif kind == "highpass":
        flt = ButterworthHighPassFilter()
        flt.cutoff_hz = float(params.get("cutoff_hz", 300.0))
        flt.order = int(params.get("order", 3))
    elif kind == "lowpass":
        flt = ButterworthLowPassFilter()
        flt.cutoff_hz = float(params.get("cutoff_hz", 3000.0))
        flt.order = int(params.get("order", 3))
    elif kind == "bandpass":
        flt = ButterworthBandPassFilter()
        flt.lowcut_hz = float(params.get("lowcut_hz", 300.0))
        flt.highcut_hz = float(params.get("highcut_hz", 3000.0))
        flt.order = int(params.get("order", 3))
    else:
        return None
    flt.enabled = True
    if hasattr(flt, "sos_cache"):
        flt.sos_cache = {}
    return flt


def apply_filter_step(matrix: np.ndarray, sample_rate: float, step: PreprocessingStep) -> np.ndarray:
    flt = _filter_from_step(step)
    if flt is None:
        return np.asarray(matrix, dtype=np.float64).copy()
    x = np.asarray(matrix, dtype=np.float64)
    return np.vstack([flt.apply(row, float(sample_rate)) for row in x])


def apply_cmr(matrix: np.ndarray, method: str = "median", exclude_rows: Optional[Sequence[int]] = None) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return x.copy()
    exclude = set(int(i) for i in (exclude_rows or []))
    keep = [i for i in range(x.shape[0]) if i not in exclude]
    if not keep:
        return x.copy()
    ref = np.mean(x[keep], axis=0) if method.lower() == "mean" else np.median(x[keep], axis=0)
    return x - ref[None, :]


def _merge_windows(windows: Sequence[Tuple[int, int]], gap: int = 0) -> List[Tuple[int, int]]:
    clean = sorted((max(0, int(a)), max(0, int(b))) for a, b in windows if int(b) > int(a))
    if not clean:
        return []
    out = [list(clean[0])]
    for a, b in clean[1:]:
        if a <= out[-1][1] + int(gap):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(int(a), int(b)) for a, b in out]


def detect_artifact_windows(
    matrix: np.ndarray,
    sample_rate: float,
    threshold_z: float = 5.0,
    min_distance_ms: float = 20.0,
    pre_ms: float = 5.0,
    post_ms: float = 25.0,
    merge_gap_ms: float = 5.0,
) -> List[Tuple[int, int]]:
    from scipy.signal import find_peaks

    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        return []
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True)
    z = np.abs((x - med) / (mad / 0.6745 + 1e-8))
    aggregate = np.median(z, axis=0)
    distance = max(1, int(round(float(min_distance_ms) * float(sample_rate) / 1000.0)))
    peaks, _props = find_peaks(aggregate, height=float(threshold_z), distance=distance)
    pre = max(0, int(round(float(pre_ms) * float(sample_rate) / 1000.0)))
    post = max(0, int(round(float(post_ms) * float(sample_rate) / 1000.0)))
    n = x.shape[1]
    gap = max(0, int(round(float(merge_gap_ms) * float(sample_rate) / 1000.0)))
    return _merge_windows([(max(0, int(pk) - pre), min(n, int(pk) + post + 1)) for pk in peaks], gap=gap)


def blank_artifact_windows(matrix: np.ndarray, windows: Sequence[Tuple[int, int]]) -> np.ndarray:
    x = np.asarray(matrix, dtype=np.float64)
    out = x.copy()
    for row_idx in range(out.shape[0]):
        row = out[row_idx]
        for a, b in windows:
            a = max(0, int(a))
            b = min(row.size, int(b))
            if b <= a:
                continue
            left = a - 1
            right = b
            if left < 0 and right >= row.size:
                row[a:b] = 0.0
            elif left < 0:
                row[a:b] = row[right]
            elif right >= row.size:
                row[a:b] = row[left]
            else:
                t = (np.arange(a, b) - left) / float(max(1, right - left))
                row[a:b] = (1.0 - t) * row[left] + t * row[right]
    return out


def apply_artifact_removal(matrix: np.ndarray, sample_rate: float, step: PreprocessingStep) -> np.ndarray:
    params = dict(step.params or {})
    windows = detect_artifact_windows(
        matrix=matrix,
        sample_rate=sample_rate,
        threshold_z=float(params.get("threshold_z", 5.0)),
        min_distance_ms=float(params.get("min_distance_ms", 20.0)),
        pre_ms=float(params.get("pre_ms", 5.0)),
        post_ms=float(params.get("post_ms", 25.0)),
        merge_gap_ms=float(params.get("merge_gap_ms", 5.0)),
    )
    return blank_artifact_windows(matrix, windows)


def apply_preprocessing_pipeline(matrix: np.ndarray, sample_rate: float, pipeline: Optional[PipelineSpec]) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    if pipeline is None:
        return out
    for step in pipeline.steps:
        if not step.enabled:
            continue
        kind = step.kind.strip().lower()
        if kind in {"notch", "highpass", "lowpass", "bandpass"}:
            out = apply_filter_step(out, sample_rate, step)
        elif kind == "cmr":
            out = apply_cmr(
                out,
                method=str(step.params.get("method", "median")),
                exclude_rows=step.params.get("exclude_rows", []),
            )
        elif kind == "artifact_removal":
            out = apply_artifact_removal(out, sample_rate, step)
    return out


__all__ = [
    "PreprocessingStep",
    "PipelineSpec",
    "apply_artifact_removal",
    "apply_cmr",
    "apply_filter_step",
    "apply_preprocessing_pipeline",
    "blank_artifact_windows",
    "default_pipeline_store",
    "detect_artifact_windows",
    "pipelines_file",
    "read_pipeline_store",
    "write_pipeline_store",
]
