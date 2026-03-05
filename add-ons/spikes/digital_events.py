from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from scipy.signal import find_peaks

from weegit.core.conversions.add_ons import BaseAddOn


class DigitalEventsAddOn(BaseAddOn):
    """
    Detects digital events on analogue input channels (e.g. TTL lines),
    using both a height threshold and minimal distance between peaks.
    """

    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header) -> Optional[Tuple[int, str, float, float]]:
        user_session = session_manager.user_session
        analogue_channels: List[int] = list(user_session.analogue_input_channel_indexes or [])
        if not analogue_channels:
            QMessageBox.warning(None, "Digital events", "No analogue input channels configured.")
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Digital events detection")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()

        # Channel selector
        channel_combo = QComboBox()
        for ch in analogue_channels:
            name = ""
            try:
                if 0 <= ch < len(header.channel_info.name):
                    name = header.channel_info.name[ch]
            except Exception:
                name = ""
            label = f"{ch}" if not name else f"{ch} [{name}]"
            channel_combo.addItem(label, ch)
        form.addRow("Digital channel:", channel_combo)

        # Event name
        name_edit = QLineEdit()
        form.addRow("Event name:", name_edit)

        # Height (can be negative)
        height_spin = QDoubleSpinBox()
        height_spin.setDecimals(6)
        height_spin.setRange(-1e9, 1e9)
        height_spin.setValue(50.0)
        form.addRow("Height (threshold):", height_spin)

        # Minimal distance between events in ms
        distance_spin = QDoubleSpinBox()
        distance_spin.setDecimals(3)
        distance_spin.setRange(0.0, 1e9)
        distance_spin.setValue(500.0)
        form.addRow("Min distance, ms:", distance_spin)

        layout.addLayout(form)

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

        event_name = name_edit.text().strip()
        if not event_name:
            QMessageBox.warning(dialog, "Digital events", "Event name must not be empty.")
            return None

        # Ensure uniqueness vs existing vocabulary
        if event_name in (session_manager.events_vocabulary or {}).values():
            QMessageBox.warning(dialog, "Digital events", "Event name must be unique.")
            return None

        channel_idx = int(channel_combo.currentData())
        height = float(height_spin.value())
        distance_ms = float(distance_spin.value())
        return channel_idx, event_name, height, distance_ms

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        experiment_data = session_manager.experiment_data
        gui_setup = session_manager.gui_setup

        params = self._ask_parameters(session_manager, header)
        if params is None:
            return

        channel_idx, event_name, height, distance_ms = params

        # Register new event vocabulary
        event_name_id = session_manager.add_event_vocabulary(event_name)

        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        if add_on_data_dir is not None:
            add_on_data_dir.mkdir(parents=True, exist_ok=True)

        total_sweeps = header.number_of_sweeps
        if total_sweeps <= 0:
            yield {"progress": 100, "message": "No sweeps to process"}
            return

        yield {"progress": 0, "message": "Detecting digital events..."}

        # Convert distance from ms to samples
        distance_samples = int((distance_ms * header.sample_rate) / 1000.0) if distance_ms > 0 else 1
        distance_samples = max(distance_samples, 1)

        for sweep_idx in range(total_sweeps):
            # Full sweep, filtered
            start_sample = 0
            end_sample = header.number_of_points_per_sweep
            output_number_of_dots = header.number_of_points_per_sweep
            each_point = 1

            signal = experiment_data.process_single_channel(
                channel_idx=channel_idx,
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=each_point,
                sample_rate=header.sample_rate,
                filters=gui_setup.filters,
                output_number_of_dots=output_number_of_dots,
                apply_filters=True,
                transformation_add_ons=[],
            )

            # If height < 0, invert signal so that peaks become positive
            search_signal = signal.copy()
            search_height = height
            if height < 0:
                search_signal = -search_signal
                search_height = -height

            peaks, props = find_peaks(search_signal, height=search_height, distance=distance_samples)
            if peaks.size == 0:
                progress = int(((sweep_idx + 1) / total_sweeps) * 100)
                yield {
                    "progress": progress,
                    "message": f"Sweep {sweep_idx + 1}/{total_sweeps}: no events",
                }
                continue

            events_specs: List[Tuple[int, int, float]] = []
            for peak_sample in peaks:
                time_ms = (peak_sample * 1000.0) / header.sample_rate
                events_specs.append((event_name_id, sweep_idx, float(time_ms)))

            if events_specs:
                session_manager.add_events(events_specs)

            progress = int(((sweep_idx + 1) / total_sweeps) * 100)
            yield {
                "progress": progress,
                "message": f"Sweep {sweep_idx + 1}/{total_sweeps} processed",
            }
