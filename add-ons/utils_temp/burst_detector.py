from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from nir_common import (
    BurstInterval,
    BurstsPayload,
    NIRBaseAddOn,
    detect_burst_intervals_from_spikes,
)


class BurstDetectorAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[Path, List[int], int, str, float, int, float]]:
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_source_dir = self.choose_threshold_dir_dialog(
            "Burst detector",
            threshold_dirs,
            selected_dir=self.state.selected_detector_threshold_dir,
            label="Source spikes:",
        )
        if selected_source_dir is None:
            return None

        groups = self.ensure_non_aux_groups(session_manager, "Burst detector")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Burst detector")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        method_combo = QComboBox()
        method_combo.addItems(["max_interval", "log_isi"])
        method_idx = method_combo.findText(str(self.state.burst_method))
        method_combo.setCurrentIndex(max(0, method_idx))
        form.addRow("Method:", method_combo)

        max_isi_spin = QDoubleSpinBox()
        max_isi_spin.setRange(1.0, 10_000.0)
        max_isi_spin.setDecimals(3)
        max_isi_spin.setValue(float(self.state.burst_max_isi_ms))
        form.addRow("Max ISI (ms):", max_isi_spin)

        min_spikes_spin = QSpinBox()
        min_spikes_spin.setRange(2, 500)
        min_spikes_spin.setValue(int(self.state.burst_min_spikes))
        form.addRow("Min spikes:", min_spikes_spin)

        min_duration_spin = QDoubleSpinBox()
        min_duration_spin.setRange(0.0, 10_000.0)
        min_duration_spin.setDecimals(3)
        min_duration_spin.setValue(float(self.state.burst_min_duration_ms))
        form.addRow("Min burst duration (ms):", min_duration_spin)

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

        channels = self.selected_channels(channels_list)
        if not channels:
            QMessageBox.warning(dialog, "Burst detector", "Select at least one channel.")
            return None

        method = method_combo.currentText().strip()
        max_isi_ms = float(max_isi_spin.value())
        min_spikes = int(min_spikes_spin.value())
        min_duration_ms = float(min_duration_spin.value())
        self.state.selected_detector_threshold_dir = str(selected_source_dir)
        self.state.burst_method = method
        self.state.burst_max_isi_ms = max_isi_ms
        self.state.burst_min_spikes = min_spikes
        self.state.burst_min_duration_ms = min_duration_ms
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=self.state.ignore_event_names,
            ignore_before_ms=self.state.ignore_before_ms,
            ignore_after_ms=self.state.ignore_after_ms,
        )
        return (
            selected_source_dir,
            channels,
            int(group_combo.currentData()),
            method,
            max_isi_ms,
            min_spikes,
            min_duration_ms,
        )

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        source_threshold_dir, channels, _group_idx, method, max_isi_ms, min_spikes, min_duration_ms = params
        self.state.selected_burst_threshold_dir = str(Path(add_on_data_dir) / source_threshold_dir.name)

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        spikes_payload = self.read_spikes_payload(source_threshold_dir, sweep_idx)
        if spikes_payload is None:
            QMessageBox.warning(None, "Burst detector", "No spikes payload for current sweep.")
            return

        bursts_by_channel = {}
        total = len(channels)
        yield {"progress": 0, "message": "Detecting channel bursts..."}
        for i, channel_idx in enumerate(channels, start=1):
            spikes = spikes_payload.spikes_by_channel.get(int(channel_idx), [])
            times_s = np.asarray([float(spike.time_ms) / 1000.0 for spike in spikes], dtype=float)
            intervals = detect_burst_intervals_from_spikes(
                times_s,
                method=method,
                max_isi_ms=max_isi_ms,
                min_spikes=min_spikes,
                min_duration_ms=min_duration_ms,
            )
            bursts_by_channel[int(channel_idx)] = [
                BurstInterval(
                    start_ms=float(times_s[start_idx] * 1000.0),
                    end_ms=float(times_s[end_idx] * 1000.0),
                    n_spikes=int(end_idx - start_idx + 1),
                )
                for start_idx, end_idx in intervals
            ]
            yield {"progress": int((i / total) * 100), "message": f"Processed channel {i}/{total}"}

        payload = BurstsPayload(
            sweep_idx=sweep_idx,
            source_threshold_dir=str(source_threshold_dir),
            method=method,
            max_isi_ms=max_isi_ms,
            min_spikes=min_spikes,
            min_duration_ms=min_duration_ms,
            bursts_by_channel=bursts_by_channel,
        )
        out_dir = Path(add_on_data_dir) / source_threshold_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sweep_idx}.bursts.json"
        out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        yield {"progress": 100, "message": f"Saved bursts: {out_path.name}"}
