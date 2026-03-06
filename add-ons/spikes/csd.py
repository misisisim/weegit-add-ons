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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QColor, QPen, QPainter, QImage
from scipy.interpolate import RectBivariateSpline

from weegit.core.add_ons.base import BaseAddOn


class CSDAddOn(BaseAddOn):
    """
    View-only add-on that draws interpolated CSD as a colored background
    behind EEG traces. Scale (vmin, vmax) is configurable and cached.
    """

    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 50
    DEFAULT_X_PIXELS_STEP = 4
    DEFAULT_Y_PIXELS_STEP = 8

    def __init__(self):
        # Cached scale configuration
        self._scale_min: Optional[float] = None
        self._scale_max: Optional[float] = None
        self._auto_scale_min: Optional[float] = None
        self._auto_scale_max: Optional[float] = None
        self._x_pixels_step: int = self.DEFAULT_X_PIXELS_STEP
        self._y_pixels_step: int = self.DEFAULT_Y_PIXELS_STEP
        self._config_loaded_for_dir: Optional[Path] = None

        # Cached CSD and rendered assets
        self._cached_csd: Optional[np.ndarray] = None
        self._cached_csd_key: Optional[Tuple] = None
        self._cached_image: Optional[QImage] = None
        self._cached_key: Optional[Tuple] = None
        self._cached_colorbar_image: Optional[QImage] = None
        self._cached_colorbar_key: Optional[Tuple[int, int]] = None

    def _invalidate_render_cache(self) -> None:
        self._cached_csd = None
        self._cached_csd_key = None
        self._cached_image = None
        self._cached_key = None

    def _clear_config(self, add_on_data_dir: Optional[Path]) -> None:
        cfg_path = self._config_path(add_on_data_dir)
        if cfg_path is None:
            return
        try:
            cfg_path.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _sanitize_pixels_step(
        value: Any, fallback: int, min_value: int = 1, max_value: int = 128
    ) -> int:
        try:
            parsed = int(value)
        except Exception:
            return int(fallback)
        return max(min_value, min(max_value, parsed))

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
        self._x_pixels_step = self.DEFAULT_X_PIXELS_STEP
        self._y_pixels_step = self.DEFAULT_Y_PIXELS_STEP
        self._config_loaded_for_dir = add_on_data_dir

        if cfg_path and cfg_path.exists():
            try:
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                if data.get("vmin") is not None and data.get("vmax") is not None:
                    self._scale_min = float(data.get("vmin"))
                    self._scale_max = float(data.get("vmax"))
                self._x_pixels_step = self._sanitize_pixels_step(
                    data.get("x_pixels_step"), self.DEFAULT_X_PIXELS_STEP
                )
                self._y_pixels_step = self._sanitize_pixels_step(
                    data.get("y_pixels_step"), self.DEFAULT_Y_PIXELS_STEP
                )
            except Exception:
                self._scale_min = None
                self._scale_max = None
                self._x_pixels_step = self.DEFAULT_X_PIXELS_STEP
                self._y_pixels_step = self.DEFAULT_Y_PIXELS_STEP

    def _save_config(self, add_on_data_dir: Optional[Path]) -> None:
        if add_on_data_dir is None:
            return
        add_on_data_dir = Path(add_on_data_dir)
        add_on_data_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = self._config_path(add_on_data_dir)
        if cfg_path is None:
            return
        payload: Dict[str, Any] = {
            "x_pixels_step": int(self._x_pixels_step),
            "y_pixels_step": int(self._y_pixels_step),
        }
        if self._scale_min is not None and self._scale_max is not None:
            payload["vmin"] = float(self._scale_min)
            payload["vmax"] = float(self._scale_max)
        try:
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    # ---- Run: choose background scale ----
    def _ask_background_scale(
        self,
        add_on_data_dir: Optional[Path],
        default_min: float,
        default_max: float,
        default_x_pixels_step: int,
        default_y_pixels_step: int,
    ) -> Optional[Tuple[Optional[float], Optional[float], bool, int, int]]:
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

        row_x_step = QHBoxLayout()
        row_x_step.addWidget(QLabel("X pixel step:"))
        spin_x_step = QSpinBox()
        spin_x_step.setRange(1, 128)
        spin_x_step.setValue(int(default_x_pixels_step))
        row_x_step.addWidget(spin_x_step)
        layout.addLayout(row_x_step)

        row_y_step = QHBoxLayout()
        row_y_step.addWidget(QLabel("Y pixel step:"))
        spin_y_step = QSpinBox()
        spin_y_step.setRange(1, 128)
        spin_y_step.setValue(int(default_y_pixels_step))
        row_y_step.addWidget(spin_y_step)
        layout.addLayout(row_y_step)

        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_reset = QPushButton("Reset to auto")
        btn_ok = QPushButton("Apply")
        actions.addStretch(1)
        actions.addWidget(btn_reset)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)

        reset_selected = {"value": False}

        def _on_reset() -> None:
            reset_selected["value"] = True
            dialog.accept()

        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)
        btn_reset.clicked.connect(_on_reset)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        x_pixels_step = int(spin_x_step.value())
        y_pixels_step = int(spin_y_step.value())
        if reset_selected["value"]:
            return None, None, True, x_pixels_step, y_pixels_step

        vmin = float(spin_min.value())
        vmax = float(spin_max.value())
        if vmin == vmax:
            # Avoid degenerate scale
            epsilon = max(1e-9, abs(vmin) * 1e-6)
            vmin -= epsilon
            vmax += epsilon
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return vmin, vmax, False, x_pixels_step, y_pixels_step

    def run(self, session_manager, add_on_data_dir):
        # CSD is computed on-the-fly in view; run() is used only
        # to let the user choose and persist background scale.
        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        self._load_config_if_needed(add_on_data_dir)

        # Defaults for the dialog:
        if self._scale_min is not None and self._scale_max is not None:
            default_min = self._scale_min
            default_max = self._scale_max
        elif self._auto_scale_min is not None and self._auto_scale_max is not None:
            default_min = self._auto_scale_min
            default_max = self._auto_scale_max
        else:
            default_min = -1.0
            default_max = 1.0

        result = self._ask_background_scale(
            add_on_data_dir,
            default_min,
            default_max,
            self._x_pixels_step,
            self._y_pixels_step,
        )
        if result is None:
            return

        vmin, vmax, reset_to_auto, x_pixels_step, y_pixels_step = result
        self._x_pixels_step = self._sanitize_pixels_step(
            x_pixels_step, self.DEFAULT_X_PIXELS_STEP
        )
        self._y_pixels_step = self._sanitize_pixels_step(
            y_pixels_step, self.DEFAULT_Y_PIXELS_STEP
        )
        if reset_to_auto:
            self._scale_min = None
            self._scale_max = None
            self._save_config(add_on_data_dir)
            self._invalidate_render_cache()
            yield {"progress": 100, "message": "CSD scale reset to auto"}
            return

        self._scale_min = vmin
        self._scale_max = vmax
        self._save_config(add_on_data_dir)
        self._invalidate_render_cache()
        # No long-running processing here, just one immediate "progress"
        yield {"progress": 100, "message": "CSD scale updated"}

    # ---- Fast CSD computation ----
    @staticmethod
    def _compute_interp_csd(
        raw_data: np.ndarray,
        signal_width: int,
        x_pixels_step: int,
        area_height: int,
        y_pixels_step: int,
    ) -> np.ndarray:
        """
        Compute interpolated CSD from data of shape (time, channels).
        Vectorized and uses RectBivariateSpline for smooth interpolation.
        """
        if raw_data.ndim != 2:
            return np.empty((0, 0), dtype=np.float32)

        n_time, n_channels = raw_data.shape
        if n_channels < 3 or n_time < 2:
            return np.empty((0, 0), dtype=np.float32)

        # Mean subtraction is done per-channel (axis=0 along time).
        data = raw_data.astype(np.float32, copy=False)
        data = data - data.mean(axis=0, keepdims=True)

        # Second spatial derivative along channels (delta_x = 1)
        csd = data[:, 2:] - 2.0 * data[:, 1:-1] + data[:, :-2]
        csd = -csd  # conventional sign inversion. Ignore / (delta_x ^ 2) because delta_x = 1

        if csd.shape[0] < 2 or csd.shape[1] < 2:
            return csd.astype(np.float32, copy=False)

        x = np.arange(csd.shape[0], dtype=np.float32)
        y = np.arange(csd.shape[1], dtype=np.float32)
        out_w = max(2, signal_width // max(1, x_pixels_step))
        out_h = max(2, area_height // max(1, y_pixels_step))
        kx = min(3, csd.shape[0] - 1)
        ky = min(3, csd.shape[1] - 1)

        try:
            spline = RectBivariateSpline(x, y, csd, kx=kx, ky=ky)
        except Exception:
            return csd.astype(np.float32, copy=False)

        x_new = np.linspace(x[0], x[-1], out_w)
        y_new = np.linspace(y[0], y[-1], out_h)
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
    def _jet_color(t: np.ndarray) -> np.ndarray:
        """
        Jet-like colormap: deep blue -> cyan -> green -> yellow -> deep red.
        t in [0, 1]. Returns array of shape (..., 4) in uint8 ARGB (B,G,R,A).
        """
        t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
        r = np.zeros_like(t, dtype=np.float32)
        g = np.zeros_like(t, dtype=np.float32)
        b = np.zeros_like(t, dtype=np.float32)
        # Segments (like matplotlib jet): 0 -> 0.125 -> 0.375 -> 0.625 -> 0.875 -> 1
        # deep blue -> blue -> cyan -> green -> yellow -> red
        m1 = t <= 0.125
        m2 = (t > 0.125) & (t <= 0.375)
        m3 = (t > 0.375) & (t <= 0.625)
        m4 = (t > 0.625) & (t <= 0.875)
        m5 = t > 0.875

        if np.any(m1):
            s = t[m1] / 0.125
            b[m1] = 128.0 + (255.0 - 128.0) * s
        if np.any(m2):
            s = (t[m2] - 0.125) / 0.25
            g[m2] = 255.0 * s
            b[m2] = 255.0
        if np.any(m3):
            s = (t[m3] - 0.375) / 0.25
            g[m3] = 255.0
            b[m3] = 255.0 * (1.0 - s)
        if np.any(m4):
            s = (t[m4] - 0.625) / 0.25
            r[m4] = 255.0 * s
            g[m4] = 255.0
        if np.any(m5):
            s = (t[m5] - 0.875) / 0.125
            r[m5] = 255.0
            g[m5] = 255.0 * (1.0 - s)

        r_u = np.clip(np.round(r), 0, 255).astype(np.uint8)
        g_u = np.clip(np.round(g), 0, 255).astype(np.uint8)
        b_u = np.clip(np.round(b), 0, 255).astype(np.uint8)
        a = np.full_like(r_u, 255, dtype=np.uint8)
        return np.stack([b_u, g_u, r_u, a], axis=-1)

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

        # Build data matrix (time, channels) from currently visible EEG channels.
        # Only visible time window is used to keep view() fast on redraw.
        series_list: List[np.ndarray] = []
        for ch_idx in digital_visible_channel_indexes:
            ch_data = processed_data.get(ch_idx)
            if ch_data is None:
                return
            series_list.append(ch_data)

        if not series_list:
            return

        lengths = {len(s) for s in series_list}
        if len(lengths) != 1:
            return

        n_samples = int(lengths.pop())
        if n_samples < 2:
            return

        data_signature: List[Tuple] = []
        for series in series_list:
            if series.size < 2:
                return
            mid = int(series.size // 2)
            data_signature.append(
                (
                    id(series),
                    int(series.shape[0]),
                    float(series[0]),
                    float(series[mid]),
                    float(series[-1]),
                )
            )

        csd_key = (
            tuple(digital_visible_channel_indexes),
            signal_width,
            eeg_area_height,
            int(self._x_pixels_step),
            int(self._y_pixels_step),
            tuple(data_signature),
        )
        if self._cached_csd is None or self._cached_csd_key != csd_key:
            data = np.stack(
                [s for s in series_list],
                axis=1,
            )
            self._cached_csd = self._compute_interp_csd(
                data,
                signal_width,
                self._x_pixels_step,
                eeg_area_height,
                self._y_pixels_step,
            )
            self._cached_csd_key = csd_key

        csd = self._cached_csd
        if csd is None:
            return
        if csd.ndim != 2 or csd.size == 0:
            return

        # Auto scale is updated from current CSD; manual scale, if set, has priority.
        csd_min = float(np.min(csd))
        csd_max = float(np.max(csd))
        if csd_max == csd_min:
            csd_max = csd_min + 1e-6
        self._auto_scale_min = csd_min
        self._auto_scale_max = csd_max

        vmin = self._scale_min if self._scale_min is not None else self._auto_scale_min
        vmax = self._scale_max if self._scale_max is not None else self._auto_scale_max
        if vmin is None or vmax is None:
            return

        # Cache key for already-rendered heatmap
        key = (
            csd_key,
            signal_width,
            eeg_area_height,
            float(vmin),
            float(vmax),
            csd.shape,
        )

        if self._cached_image is None or self._cached_key != key:
            norm = self._normalize(csd.T, vmin, vmax)
            argb = self._jet_color(norm)

            h, w, _ = argb.shape
            # QImage expects bytes in row-major order
            self._cached_image = QImage(
                argb.tobytes(), w, h, QImage.Format.Format_ARGB32
            ).copy()
            self._cached_key = key

        if self._cached_image is None or self._cached_image.isNull():
            return

        # Draw CSD background into the same area where EEG traces are drawn
        left_margin = 0  # getattr(signal_widget, "_left_margin", 0)
        rect = QRect(left_margin, 0, signal_width, eeg_area_height)
        painter.drawImage(rect, self._cached_image)

        # Draw compact legend in bottom-right on semi-transparent background
        bar_width = max(8, int(signal_width * 0.012))
        bar_height = max(28, int(eeg_area_height * 0.22))
        panel_padding = 6
        label_width = 46
        title_height = 14
        panel_width = bar_width + label_width + panel_padding * 3
        panel_height = bar_height + title_height + panel_padding * 2
        panel_x = left_margin + signal_width - panel_width - 6
        panel_y = eeg_area_height - panel_height - 6

        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 70))
        painter.drawRoundedRect(panel_x, panel_y, panel_width, panel_height, 5, 5)
        painter.restore()

        bar_x = panel_x + panel_padding + label_width
        bar_y = panel_y + panel_padding + title_height

        bar_key = (bar_width, bar_height)
        if self._cached_colorbar_image is None or self._cached_colorbar_key != bar_key:
            grad_steps = bar_height
            # Top = vmax (red), bottom = vmin (blue): reverse t
            t = np.linspace(1.0, 0.0, grad_steps, dtype=np.float32)[:, None]
            argb_bar = self._jet_color(t)
            bar_img = QImage(
                argb_bar.tobytes(), 1, grad_steps, QImage.Format.Format_ARGB32
            ).scaled(bar_width, bar_height)
            self._cached_colorbar_image = bar_img.copy()
            self._cached_colorbar_key = bar_key
        painter.drawImage(
            QRect(bar_x, bar_y, bar_width, bar_height), self._cached_colorbar_image
        )

        # Colorbar labels and title
        painter.setPen(QPen(text_color))
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(
            QRect(panel_x + panel_padding, panel_y + 1, panel_width - panel_padding * 2, title_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "CSD",
        )

        # Always render five scale labels.
        values: List[float] = [vmax, 0.5 * vmax, 0.0, 0.5 * vmin, vmin]
        for val in values:
            if vmax == vmin:
                t_val = 0.5
            else:
                t_val = (val - vmin) / (vmax - vmin)
            t_val = max(0.0, min(1.0, t_val))
            # y coordinate on the bar: t=1 at top, t=0 at bottom
            y_center = bar_y + int((1.0 - t_val) * (bar_height - 1))
            painter.drawText(
                QRect(panel_x + panel_padding, y_center - 7, label_width, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{val:.1f}",
            )
