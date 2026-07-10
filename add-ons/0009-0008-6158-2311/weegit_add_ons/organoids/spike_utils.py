from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks


@dataclass
class SpikePoint:
    sample_idx: int
    time_ms: float
    value_uv: float
    polarity: str = "negative"


def rolling_sigma_mad(
    signal_1d: np.ndarray,
    fs: float,
    window_ms: float = 500.0,
    step_ms: float = 100.0,
    sigma_floor_uv: float = 2.0,
    smooth_windows: int = 3,
    startup_lock_ms: Optional[float] = None,
    end_lock_ms: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    min_valid_fraction: float = 0.20,
) -> np.ndarray:
    x = np.asarray(signal_1d, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.array([], dtype=np.float64)
    valid_mask = None
    if mask is not None:
        valid_mask = np.asarray(mask, dtype=bool)
        if valid_mask.size != n:
            valid_mask = None
    w = min(n, max(8, int(round(float(window_ms) * float(fs) / 1000.0))))
    h = max(1, int(round(float(step_ms) * float(fs) / 1000.0)))
    centers = []
    sigma_vals = []
    prev_sigma = float(sigma_floor_uv)
    min_valid_n = max(8, int(round(float(min_valid_fraction) * w)))
    for start in range(0, max(1, n - w + 1), h):
        seg = x[start:start + w]
        seg_valid = seg[valid_mask[start:start + w]] if valid_mask is not None else seg
        if seg_valid.size >= min_valid_n:
            med = np.median(seg_valid)
            mad = np.median(np.abs(seg_valid - med))
            sigma = mad / 0.6745 if mad > 0 else 0.0
            prev_sigma = max(float(sigma), float(sigma_floor_uv))
        centers.append(start + w // 2)
        sigma_vals.append(prev_sigma)
    if not centers:
        return np.full(n, float(sigma_floor_uv), dtype=np.float64)
    sigma_vals_arr = np.asarray(sigma_vals, dtype=np.float64)
    if int(smooth_windows) > 1 and sigma_vals_arr.size > 1:
        k = int(max(1, smooth_windows))
        sigma_vals_arr = np.convolve(sigma_vals_arr, np.ones(k) / float(k), mode="same")
    sigma_full = np.interp(
        np.arange(n, dtype=np.float64),
        np.asarray(centers, dtype=np.float64),
        sigma_vals_arr,
        left=sigma_vals_arr[0],
        right=sigma_vals_arr[-1],
    )
    global_sigma = max(float(sigma_floor_uv), float(np.median(sigma_vals_arr)))
    startup_n = int(round(float(startup_lock_ms) * fs / 1000.0)) if startup_lock_ms is not None else int(round(w / 2))
    startup_n = max(0, min(startup_n, n))
    if startup_n > 0:
        sigma_full[:startup_n] = np.maximum(sigma_full[:startup_n], global_sigma)
    end_n = int(round(float(end_lock_ms) * fs / 1000.0)) if end_lock_ms is not None else startup_n
    end_n = max(0, min(end_n, n))
    if end_n > 0:
        sigma_full[n - end_n:] = np.maximum(sigma_full[n - end_n:], global_sigma)
    return sigma_full


def mad_sigma(signal_1d: np.ndarray, sigma_floor_uv: float = 0.0) -> float:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size == 0:
        return float(sigma_floor_uv)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(float(sigma_floor_uv), float(mad / 0.6745 if mad > 0 else 0.0))


def _spikes_from_indices(signal_1d: np.ndarray, peak_indices: Sequence[int], fs: float, polarity: str) -> list[SpikePoint]:
    x = np.asarray(signal_1d, dtype=np.float64)
    out: list[SpikePoint] = []
    for idx in np.asarray(peak_indices, dtype=np.int64):
        if 0 <= int(idx) < x.size:
            out.append(
                SpikePoint(
                    sample_idx=int(idx),
                    time_ms=float(int(idx) * 1000.0 / float(fs)),
                    value_uv=float(x[int(idx)]),
                    polarity=polarity,
                )
            )
    return out


def detect_spikes_mad(
    signal_1d: np.ndarray,
    fs: float,
    multiplier: float = 6.0,
    min_distance_ms: float = 1.0,
    detect_positive: bool = False,
    sigma_floor_uv: float = 0.0,
) -> list[SpikePoint]:
    x = np.asarray(signal_1d, dtype=np.float64)
    sigma = mad_sigma(x, sigma_floor_uv=sigma_floor_uv)
    threshold = float(multiplier) * sigma
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    if detect_positive:
        peaks, _props = find_peaks(x, height=threshold, distance=distance)
        return _spikes_from_indices(x, peaks, fs, "positive")
    peaks, _props = find_peaks(-x, height=threshold, distance=distance)
    return _spikes_from_indices(x, peaks, fs, "negative")


def detect_spikes_adaptive_mad(
    signal_1d: np.ndarray,
    fs: float,
    multiplier: float = 6.0,
    min_distance_ms: float = 1.0,
    detect_positive: bool = False,
    sigma_t: Optional[np.ndarray] = None,
    sigma_floor_uv: float = 2.0,
    ignore_initial_ms: float = 0.0,
    ignore_final_ms: float = 0.0,
) -> list[SpikePoint]:
    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size == 0:
        return []
    sigma_arr = rolling_sigma_mad(x, fs, sigma_floor_uv=sigma_floor_uv) if sigma_t is None else np.asarray(sigma_t, dtype=np.float64)
    if sigma_arr.size != x.size:
        sigma_arr = np.full(x.size, mad_sigma(x, sigma_floor_uv=sigma_floor_uv), dtype=np.float64)
    thr = float(multiplier) * sigma_arr
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    mask = x > thr if detect_positive else (-x > thr)
    candidate = np.flatnonzero(mask)
    start_n = max(0, int(round(float(ignore_initial_ms) * float(fs) / 1000.0)))
    end_n = max(0, int(round(float(ignore_final_ms) * float(fs) / 1000.0)))
    if start_n > 0:
        candidate = candidate[candidate >= start_n]
    if end_n > 0:
        candidate = candidate[candidate < (x.size - end_n)]
    if candidate.size == 0:
        return []
    picked = []
    best_idx = int(candidate[0])
    best_score = float((x[best_idx] - thr[best_idx]) if detect_positive else (-x[best_idx] - thr[best_idx]))
    for raw_idx in candidate[1:]:
        idx = int(raw_idx)
        score = float((x[idx] - thr[idx]) if detect_positive else (-x[idx] - thr[idx]))
        if idx - best_idx <= distance:
            if score > best_score:
                best_idx = idx
                best_score = score
        else:
            picked.append(best_idx)
            best_idx = idx
            best_score = score
    picked.append(best_idx)
    return _spikes_from_indices(x, picked, fs, "positive" if detect_positive else "negative")


def merge_spikes_global(spikes: Sequence[SpikePoint], fs: float, min_distance_ms: float = 1.0) -> list[SpikePoint]:
    rows = sorted(list(spikes), key=lambda sp: int(sp.sample_idx))
    if not rows:
        return []
    distance = max(1, int(round(float(min_distance_ms) * float(fs) / 1000.0)))
    picked: list[SpikePoint] = []
    best = rows[0]
    for sp in rows[1:]:
        if int(sp.sample_idx) - int(best.sample_idx) <= distance:
            if abs(float(sp.value_uv)) > abs(float(best.value_uv)):
                best = sp
        else:
            picked.append(best)
            best = sp
    picked.append(best)
    return picked


def spike_time_to_seconds(spikes: Optional[Sequence[object]], fs: Optional[float] = None) -> np.ndarray:
    if not spikes:
        return np.array([], dtype=np.float64)
    out = []
    for sp in spikes:
        if hasattr(sp, "time_ms"):
            out.append(float(getattr(sp, "time_ms")) / 1000.0)
        elif isinstance(sp, Mapping) and "time_ms" in sp:
            out.append(float(sp["time_ms"]) / 1000.0)
        elif fs is not None:
            out.append(float(sp) / float(fs))
    return np.asarray(out, dtype=np.float64)


def symmetric_autocorrelogram(spike_times_s: Sequence[float], fs: float, max_lag_ms: float = 100.0, bin_ms: float = 1.0):
    t = np.sort((np.asarray(spike_times_s, dtype=np.float64) * float(fs)).astype(np.int64))
    max_lag = int(round(float(max_lag_ms) / 1000.0 * float(fs)))
    bin_s = max(1, int(round(float(bin_ms) / 1000.0 * float(fs))))
    centers_s = np.arange(-max_lag, max_lag + 1, bin_s, dtype=np.int64)
    edges = np.concatenate([[centers_s[0] - bin_s / 2.0], (centers_s[:-1] + centers_s[1:]) / 2.0, [centers_s[-1] + bin_s / 2.0]])
    centers = centers_s.astype(np.float64) / float(fs) * 1000.0
    if t.size < 2:
        return centers, np.zeros_like(centers, dtype=np.float64)
    diffs = []
    for i in range(t.size - 1):
        j0 = np.searchsorted(t, t[i] + 1, side="left")
        j1 = np.searchsorted(t, t[i] + max_lag, side="right")
        if j1 > j0:
            diffs.append(t[j0:j1] - t[i])
    if not diffs:
        return centers, np.zeros_like(centers, dtype=np.float64)
    d = np.concatenate(diffs).astype(np.float64)
    hist, _edges = np.histogram(np.concatenate([d, -d]), bins=edges)
    return centers, 0.5 * (hist.astype(np.float64) + hist[::-1].astype(np.float64))


def crosscorrelogram(spike_times1_s: Sequence[float], spike_times2_s: Sequence[float], fs: float, max_lag_ms: float = 100.0, bin_ms: float = 1.0):
    t1 = np.sort((np.asarray(spike_times1_s, dtype=np.float64) * float(fs)).astype(np.int64))
    t2 = np.sort((np.asarray(spike_times2_s, dtype=np.float64) * float(fs)).astype(np.int64))
    max_lag = int(round(float(max_lag_ms) / 1000.0 * float(fs)))
    bin_s = max(1, int(round(float(bin_ms) / 1000.0 * float(fs))))
    centers_s = np.arange(-max_lag, max_lag + 1, bin_s, dtype=np.int64)
    edges = np.concatenate([[centers_s[0] - bin_s / 2.0], (centers_s[:-1] + centers_s[1:]) / 2.0, [centers_s[-1] + bin_s / 2.0]])
    centers = centers_s.astype(np.float64) / float(fs) * 1000.0
    hist = np.zeros(len(centers), dtype=np.float64)
    if t1.size == 0 or t2.size == 0:
        return centers, hist
    for t in t1:
        i0 = np.searchsorted(t2, t - max_lag, side="left")
        i1 = np.searchsorted(t2, t + max_lag, side="right")
        if i1 > i0:
            h, _edges = np.histogram((t2[i0:i1] - t).astype(np.float64), bins=edges)
            hist += h.astype(np.float64)
    return centers, hist


def extract_waves(signal_1d: np.ndarray, spike_times_s: Sequence[float], fs: float, pre_ms: float = 2.0, post_ms: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(signal_1d, dtype=np.float64)
    pre = max(1, int(round(float(pre_ms) * float(fs) / 1000.0)))
    post = max(1, int(round(float(post_ms) * float(fs) / 1000.0)))
    waves = []
    centers = []
    for t in np.asarray(spike_times_s, dtype=np.float64):
        c = int(round(float(t) * float(fs)))
        s, e = c - pre, c + post
        if s < 0 or e >= x.size:
            continue
        waves.append(x[s:e].copy())
        centers.append(c)
    if not waves:
        return np.empty((0, pre + post), dtype=np.float64), np.array([], dtype=np.int64)
    return np.asarray(waves, dtype=np.float64), np.asarray(centers, dtype=np.int64)


def make_raster_arrays(spikes_by_channel, channel_order=None):
    all_t = []
    all_r = []
    ordered = sorted(spikes_by_channel.keys()) if channel_order is None else [ch for ch in channel_order if ch in spikes_by_channel]
    for row, ch in enumerate(ordered):
        st = spike_time_to_seconds(spikes_by_channel.get(ch, []))
        if st.size:
            all_t.append(st)
            all_r.append(np.full_like(st, row, dtype=np.float64))
    if not all_t:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64), ordered
    return np.concatenate(all_t), np.concatenate(all_r), ordered


def avg_spike_csd_classic(signals_matrix, spike_times_s, fs, channel_positions_um, window_ms=4.0, smooth_sigma_ms=None):
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(signals_matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3:
        return {"t_ms": np.array([]), "avg_waves": np.empty((0, 0)), "csd": np.empty((0, 0)), "n_spikes_used": 0}
    half = int((float(window_ms) / 2.0) * float(fs) / 1000.0)
    total = 2 * half
    acc = np.zeros((x.shape[0], total), dtype=np.float64)
    kept = 0
    for c in (np.asarray(spike_times_s, dtype=np.float64) * float(fs)).astype(np.int64):
        s, e = int(c) - half, int(c) + half
        if s < 0 or e >= x.shape[1]:
            continue
        acc += x[:, s:e]
        kept += 1
    if kept > 0:
        acc /= float(kept)
    dz = float(np.mean(np.diff(np.asarray(channel_positions_um, dtype=np.float64))))
    d1 = np.gradient(acc, dz, axis=0)
    csd = -np.gradient(d1, dz, axis=0)
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        csd = gaussian_filter1d(csd, sigma=float(smooth_sigma_ms) / 1000.0 * float(fs), axis=1)
    return {
        "t_ms": np.arange(-half, half, dtype=np.float64) / float(fs) * 1000.0,
        "avg_waves": acc,
        "csd": csd,
        "n_spikes_used": int(kept),
        "ch_positions_um": np.asarray(channel_positions_um, dtype=np.float64),
    }


__all__ = [
    "SpikePoint",
    "avg_spike_csd_classic",
    "crosscorrelogram",
    "detect_spikes_adaptive_mad",
    "detect_spikes_mad",
    "extract_waves",
    "mad_sigma",
    "make_raster_arrays",
    "merge_spikes_global",
    "rolling_sigma_mad",
    "spike_time_to_seconds",
    "symmetric_autocorrelogram",
]
