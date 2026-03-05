from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QColor, QPen, QPainter, QImage
from scipy.interpolate import RectBivariateSpline

from weegit.core.conversions.add_ons import BaseAddOn


class CSDAddOn(BaseAddOn):
    """
    View-only add-on that draws interpolated CSD as a colored background
    behind EEG traces. Scale (vmin, vmax) is configurable and cached.
    """

    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 50

    def __init__(self):
        # Cached scale configuration
        self._scale_min: Optional[float] = None
        self._scale_max: Optional[float] = None
        self._config_loaded_for_dir: Optional[Path] = None

        # Cached image for the last view parameters
        self._cached_image: Optional[QImage] = None
        self._cached_key: Optional[Tuple] = None

    # ---- Config helpers ----
    def _config_path(self, add_on_data_dir: Optional[Path]) -> Optional[Path]:
        if add_on_data_dir is None:
            return None
        return Path(add_on_data_dir) / "csd_config.json"

    def _load_config_if_needed(self, add_on_data_dir: Optional[Path]) -> None:
        if add_on_data_dir is None:
            return
        add_on_data_dir = Path(add_on_data_dir)
        if self._config_loaded_for_dir == add_on_data_dir:
            return

        cfg_path = self._config_path(add_on_data_dir)
        self._scale_min = None
        self._scale_max = None
        self._config_loaded_for_dir = add_on_data_dir

        if cfg_path and cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                self._scale_min = float(data.get("vmin"))
                self._scale_max = float(data.get("vmax"))
            except Exception:
                self._scale_min = None
                self._scale_max = None

    def _save_config(self, add_on_data_dir: Optional[Path]) -> None:
        if add_on_data_dir is None:
            return
        if self._scale_min is None or self._scale_max is None:
            return
        add_on_data_dir = Path(add_on_data_dir)
        add_on_data_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self._config_path(add_on_data_dir)
        if cfg_path is None:
            return
        payload = {"vmin": float(self._scale_min), "vmax": float(self._scale_max)}
        try:
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    # ---- Run: choose background scale ----
    def _ask_background_scale(
        self, add_on_data_dir: Optional[Path], default_min: float, default_max: float
    ) -> Optional[Tuple[float, float]]:
        dialog = QDialog()
        dialog.setWindowTitle("CSD background scale")
        layout = QVBoxLayout(dialog)

        row_min = QHBoxLayout()
        row_min.addWidget(QLabel("Scale min:"))
        spin_min = QDoubleSpinBox()
        spin_min.setRange(-1e9, 1e9)
        spin_min.setDecimals(6)
        spin_min.setValue(float(default_min))
        row_min.addWidget(spin_min)
        layout.addLayout(row_min)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Scale max:"))
        spin_max = QDoubleSpinBox()
        spin_max.setRange(-1e9, 1e9)
        spin_max.setDecimals(6)
        spin_max.setValue(float(default_max))
        row_max.addWidget(spin_max)
        layout.addLayout(row_max)

        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Apply")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)

        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        vmin = float(spin_min.value())
        vmax = float(spin_max.value())
        if vmin == vmax:
            # Avoid degenerate scale
            epsilon = max(1e-9, abs(vmin) * 1e-6)
            vmin -= epsilon
            vmax += epsilon
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax

    def run(self, session_manager, add_on_data_dir):
        # CSD is computed on-the-fly in view; run() is used only
        # to let the user choose and persist background scale.
        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        self._load_config_if_needed(add_on_data_dir)

        # Fallback defaults if we still don't have config:
        default_min = self._scale_min if self._scale_min is not None else -1.0
        default_max = self._scale_max if self._scale_max is not None else 1.0

        result = self._ask_background_scale(add_on_data_dir, default_min, default_max)
        if result is None:
            return

        self._scale_min, self._scale_max = result
        self._save_config(add_on_data_dir)
        # No long-running processing here, just one immediate "progress"
        yield {"progress": 100, "message": "CSD scale updated"}

    # ---- Fast CSD computation ----
    @staticmethod
    def _compute_interp_csd(raw_data: np.ndarray, factor: int = 4) -> np.ndarray:
        """
        Compute interpolated CSD from data of shape (time, channels).
        Vectorized and uses RectBivariateSpline for smooth interpolation.
        """
        if raw_data.ndim != 2:
            return raw_data

        n_time, n_channels = raw_data.shape
        if n_channels < 3 or n_time < 2:
            return raw_data

        # Subtract mean across time (offset removal)
        data = raw_data - raw_data.mean(axis=0, keepdims=True)

        # Second spatial derivative along channels (delta_x = 1)
        csd = data[:, 2:] - 2.0 * data[:, 1:-1] + data[:, :-2]
        csd = -csd  # conventional sign inversion

        x = np.arange(n_time, dtype=float)
        y = np.arange(csd.shape[1], dtype=float)

        if n_time < 2 or csd.shape[1] < 2:
            return csd

        try:
            spline = RectBivariateSpline(x, y, csd, kx=3, ky=3)
        except Exception:
            return csd

        x_new = np.linspace(x[0], x[-1], max(n_time, 2 * factor))
        y_new = np.linspace(y[0], y[-1], max(csd.shape[1] * factor, csd.shape[1]))
        csd_interp = spline(x_new, y_new, grid=True)
        return csd_interp.astype(np.float32)

    # ---- Color helpers ----
    @staticmethod
    def _normalize(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        if vmax <= vmin:
            return np.zeros_like(data, dtype=np.float32)
        norm = (data - vmin) / (vmax - vmin)
        return np.clip(norm, 0.0, 1.0)

    @staticmethod
    def _blue_red_color(t: np.ndarray) -> np.ndarray:
        """
        Simple blue->red gradient. t in [0, 1].
        Returns array of shape (..., 4) in uint8 ARGB.
        """
        t = np.clip(t, 0.0, 1.0)
        r = (t * 255).astype(np.uint8)
        g = np.zeros_like(r, dtype=np.uint8)
        b = ((1.0 - t) * 255).astype(np.uint8)
        a = np.full_like(r, 255, dtype=np.uint8)
        return np.stack([b, g, r, a], axis=-1)  # QImage.Format_ARGB32 (B,G,R,A)

    # ---- View ----
    def view(
        self,
        add_on_data_dir: Path,
        # DATA
        processed_data: Dict[int, np.ndarray[np.float64]],
        voltage_scale: float,
        start_point: int,
        duration_ms: float,
        start_time_ms: float,
        end_time_ms: float,
        sample_rate: float,
        axis_duration_ms: float,
        sweep_idx: int,
        digital_visible_channel_indexes: List[int],
        channel_names: List[str],
        visible_events: List[Any],
        visible_periods: List[Any],
        analogue_channel_indexes: List[int],
        analogue_channels_setup: List[Any],
        analogue_panel_height: int,
        # UI
        painter: QPainter,
        signal_widget: QWidget,
        digital_channel_rects: List[Tuple[int, QRect]],
        signal_width: int,
        eeg_area_height: int,
        bg_color: QColor,
        grid_color: QColor,
        signal_color: QColor,
        text_color: QColor,
        axis_color: QColor,
    ):
        if (
            painter is None
            or signal_widget is None
            or duration_ms <= 0
            or signal_width <= 0
            or eeg_area_height <= 0
        ):
            return

        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        self._load_config_if_needed(add_on_data_dir)

        if not digital_visible_channel_indexes:
            return

        # Build data matrix (time, channels) from currently visible EEG channels
        series_list: List[np.ndarray] = []
        for ch_idx in digital_visible_channel_indexes:
            ch_data = processed_data.get(ch_idx)
            if ch_data is None:
                return
            series_list.append(np.asarray(ch_data, dtype=np.float32))

        if not series_list:
            return

        lengths = {len(s) for s in series_list}
        if len(lengths) != 1:
            return

        data = np.stack(series_list, axis=1)  # (time, channels)
        csd = self._compute_interp_csd(data)
        if csd.ndim != 2 or csd.size == 0:
            return

        # Determine or update scale from data if not yet known
        csd_min = float(np.min(csd))
        csd_max = float(np.max(csd))
        if self._scale_min is None or self._scale_max is None:
            # Initialize from current data and persist
            self._scale_min = csd_min
            self._scale_max = csd_max if csd_max != csd_min else csd_min + 1e-6
            self._save_config(add_on_data_dir)

        vmin = self._scale_min if self._scale_min is not None else csd_min
        vmax = self._scale_max if self._scale_max is not None else csd_max

        # Cache key: enough to reflect current visible state
        key = (
            signal_width,
            eeg_area_height,
            len(digital_visible_channel_indexes),
            float(start_time_ms),
            float(end_time_ms),
            tuple(digital_visible_channel_indexes),
            float(vmin),
            float(vmax),
            csd.shape,
        )

        if self._cached_image is None or self._cached_key != key:
            # Resample CSD grid to match drawing resolution
            h_src, w_src = csd.shape[1], csd.shape[0]  # (channels, time)
            if w_src <= 0 or h_src <= 0:
                return

            x_idx = np.linspace(0, w_src - 1, signal_width).astype(np.int64)
            y_idx = np.linspace(0, h_src - 1, eeg_area_height).astype(np.int64)
            csd_resampled = csd[x_idx[:, None], y_idx[None, :]]  # (signal_width, eeg_area_height)
            csd_resampled = csd_resampled.T  # (height, width)

            norm = self._normalize(csd_resampled, vmin, vmax)
            argb = self._blue_red_color(norm)

            h, w, _ = argb.shape
            # QImage expects bytes in row-major order
            self._cached_image = QImage(
                argb.data, w, h, QImage.Format.Format_ARGB32
            ).copy()
            self._cached_key = key

        if self._cached_image is None or self._cached_image.isNull():
            return

        # Draw CSD background into the same area where EEG traces are drawn
        axis_width = getattr(signal_widget, "_axis_width", signal_width)
        left_margin = getattr(signal_widget, "_left_margin", 0)
        rect = QRect(left_margin, 0, axis_width, eeg_area_height)
        painter.drawImage(rect, self._cached_image)

        # Draw simple vertical color bar with min/max labels in bottom-right corner
        bar_width = max(10, int(axis_width * 0.02))
        bar_height = max(40, int(eeg_area_height * 0.4))
        bar_x = left_margin + axis_width - bar_width - 4
        bar_y = eeg_area_height - bar_height - 4

        grad_steps = bar_height
        t = np.linspace(0.0, 1.0, grad_steps, dtype=np.float32)[:, None]
        argb_bar = self._blue_red_color(t)
        bar_img = QImage(
            argb_bar.data, 1, grad_steps, QImage.Format.Format_ARGB32
        ).scaled(bar_width, bar_height)
        painter.drawImage(QRect(bar_x, bar_y, bar_width, bar_height), bar_img)

        # Labels
        painter.setPen(QPen(text_color))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QRect(bar_x - 60, bar_y - 14, 60, 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{vmax:.3g}",
        )
        painter.drawText(
            QRect(bar_x - 60, bar_y + bar_height, 60, 14),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{vmin:.3g}",
        )