from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QWidget
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.add_on_organoids.organoids_common import NIRBaseAddOn, SpikesPayload


class SpikesViewerAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = True
    RUNNABLE = True
    Z_INDEX = 250

    def __init__(self):
        self._cached_path: Optional[Path] = None
        self._cached_mtime: Optional[float] = None
        self._cached_payload: Optional[SpikesPayload] = None

    def run(self, session_manager, add_on_data_dir):
        threshold_dirs = self.list_detector_threshold_dirs(Path(add_on_data_dir))
        selected = self.choose_threshold_dir_dialog("Spikes viewer", threshold_dirs)
        if selected is None:
            return
        self.state.selected_detector_threshold_dir = str(selected)
        yield {"progress": 100, "message": f"Selected {selected.name}"}

    def _load_payload(self, sweep_idx: int) -> Optional[SpikesPayload]:
        if not self.state.selected_detector_threshold_dir:
            return None
        path = Path(self.state.selected_detector_threshold_dir) / f"{int(sweep_idx)}.spikes.json"
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if self._cached_path == path and self._cached_mtime == mtime and self._cached_payload is not None:
            return self._cached_payload
        payload = SpikesPayload.model_validate_json(path.read_text(encoding="utf-8"))
        self._cached_path = path
        self._cached_mtime = mtime
        self._cached_payload = payload
        return payload

    def view(
        self,
        add_on_data_dir: Path,
        processed_data: Dict[int, np.ndarray[np.float64]],
        voltage_scale: float,
        start_point: int,
        duration_ms: float,
        start_time_ms: float,
        end_time_ms: float,
        sample_rate: float,
        axis_duration_ms: float,
        sweep_idx: int,
        visible_channel_indexes: List[int],
        channel_names: List[str],
        visible_events: List[Any],
        visible_periods: List[Any],
        channel_groups: List[Any],
        channels_setup: Dict[int, Any],
        painter: QPainter,
        signal_widget: QWidget,
        channel_rects: List[Tuple[int, QRect]],
        signal_width: int,
        draw_area_height: int,
        bg_color: QColor,
        grid_color: QColor,
        signal_color: QColor,
        text_color: QColor,
        axis_color: QColor,
    ):
        if painter is None or signal_widget is None or duration_ms <= 0 or axis_duration_ms <= 0:
            return
        payload = self._load_payload(int(sweep_idx))
        if payload is None:
            return
        axis_width = max(0, getattr(signal_widget, "_axis_width", 0))
        if axis_width <= 0:
            return

        pen = QPen(QColor(255, 0, 0))
        painter.setPen(pen)
        painter.setBrush(QColor(255, 0, 0))
        marker_size = 4
        half = marker_size // 2

        for channel_idx, rect in channel_rects:
            spikes = payload.spikes_by_channel.get(int(channel_idx))
            if not spikes:
                continue
            y = rect.center().y()
            for spike in spikes:
                if not (start_time_ms <= spike.time_ms <= end_time_ms):
                    continue
                x = ((spike.time_ms - start_time_ms) / axis_duration_ms) * axis_width
                painter.drawRect(int(x) - half, y - half, marker_size, marker_size)
