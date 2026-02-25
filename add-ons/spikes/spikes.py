from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QDialog, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt6.QtGui import QColor, QPen, QPainter
from scipy.signal import find_peaks

from weegit.core.weegit_session import Spike, Spikes
from weegit.core.conversions.add_ons import BaseAddOn


class SpikesAddOn(BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 250

    def __init__(self):
        self._cached_sweep_idx: Optional[int] = None
        self._cached_spikes: Optional[Spikes] = None
        self._cached_spikes_path: Optional[Path] = None
        self._cached_mtime: Optional[float] = None

    def transform(self, channel_data: np.ndarray, sample_rate: float):
        return channel_data

    def _ask_threshold(self, current_threshold: float = 5.0) -> Optional[float]:
        dialog = QDialog()
        dialog.setWindowTitle("Spikes detection")
        layout = QVBoxLayout(dialog)
        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold:"))
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1e9)
        spin.setSingleStep(0.1)
        spin.setValue(float(current_threshold))
        row.addWidget(spin)
        layout.addLayout(row)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_run = QPushButton("Run")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_run)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_run.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return float(spin.value())

    def run(self, user_session, experiment_data, add_on_data_dir):
        threshold = self._ask_threshold(5.0)
        if threshold is None:
            return

        add_ons_data_dir = Path(add_on_data_dir)
        add_ons_data_dir.mkdir(parents=True, exist_ok=True)

        gui_setup = user_session.gui_setup
        header = experiment_data.header
        sweep_idx = gui_setup.current_sweep_idx
        channel_indexes = list(user_session.eeg_channel_indexes)

        if not channel_indexes:
            yield {"progress": 100, "message": "No EEG channels to process"}
            return

        start_sample = 0
        output_number_of_dots = header.number_of_points_per_sweep
        end_sample = header.number_of_points_per_sweep

        spikes_by_channel: Dict[int, List[Spike]] = {}
        total = len(channel_indexes)
        yield {"progress": 0, "message": f"Detecting spikes for sweep {sweep_idx}..."}

        for idx, channel_idx in enumerate(channel_indexes, start=1):
            channel_data = experiment_data.process_single_channel(
                channel_idx=channel_idx,
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=1,
                sample_rate=header.sample_rate,
                filters=gui_setup.filters,
                output_number_of_dots=output_number_of_dots,
                apply_filters=True,
            )
            mad = float(np.median(np.abs(channel_data - np.median(channel_data))))
            threshold_value = threshold * mad / 0.6745 if mad > 0 else threshold
            peaks, _ = find_peaks(-channel_data, height=threshold_value, distance=int(0.001 * header.sample_rate))
            vals = channel_data[peaks]
            spikes_by_channel[channel_idx] = [
                Spike(
                    time_ms=(start_sample + peak_sample) * 1_000 / header.sample_rate,
                    value=float(val),
                )
                for peak_sample, val in zip(peaks, vals)
            ]
            progress = int((idx / total) * 100)
            yield {"progress": progress, "message": f"Processed channel {idx}/{total}"}

        spikes = Spikes(spikes_by_channel=spikes_by_channel, threshold=threshold)
        out_path = add_ons_data_dir / f"{sweep_idx}.spikes"
        out_path.write_text(spikes.model_dump_json(indent=0), encoding="utf-8")

        self._cached_sweep_idx = sweep_idx
        self._cached_spikes = spikes
        self._cached_spikes_path = out_path
        self._cached_mtime = out_path.stat().st_mtime
        yield {"progress": 100, "message": f"Saved spikes to {out_path.name}"}

    def _load_spikes_for_sweep(self, sweep_idx: int, add_ons_data_dir: Optional[Path]) -> Optional[Spikes]:
        if add_ons_data_dir is None:
            return None
        spikes_path = Path(add_ons_data_dir) / f"{sweep_idx}.spikes"
        if not spikes_path.exists():
            return None
        mtime = spikes_path.stat().st_mtime
        if (
                self._cached_sweep_idx == sweep_idx
                and self._cached_spikes is not None
                and self._cached_spikes_path == spikes_path
                and self._cached_mtime == mtime
        ):
            return self._cached_spikes
        spikes = Spikes.model_validate_json(spikes_path.read_text(encoding="utf-8"))
        self._cached_sweep_idx = sweep_idx
        self._cached_spikes = spikes
        self._cached_spikes_path = spikes_path
        self._cached_mtime = mtime
        return spikes

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
        if painter is None or signal_widget is None or duration_ms <= 0:
            return

        spikes = self._load_spikes_for_sweep(sweep_idx, add_on_data_dir)
        if spikes is None:
            return

        axis_width = max(0, getattr(signal_widget, "_axis_width", 0))
        if axis_width <= 0:
            return

        pen = QPen(QColor(255, 0, 0))
        painter.setPen(pen)
        painter.setBrush(QColor(255, 0, 0))
        size = 4
        half = size // 2

        for channel_idx, channel_rect in digital_channel_rects:
            spikes_list = spikes.spikes_by_channel.get(channel_idx)
            if not spikes_list:
                continue

            y = channel_rect.center().y()
            for spike in spikes_list:
                if not (start_time_ms <= spike.time_ms <= end_time_ms):
                    continue

                x = ((spike.time_ms - start_time_ms) / axis_duration_ms) * axis_width
                x_int = int(x)
                painter.drawRect(x_int - half, y - half, size, size)
