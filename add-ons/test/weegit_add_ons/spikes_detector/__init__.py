from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from scipy.signal import find_peaks
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.organoids_common import (
    FilterEditor,
    IgnoreEventsRule,
    NIRBaseAddOn,
    SpikePoint,
    SpikesPayload,
    build_valid_mask,
    filter_from_spec,
    safe_threshold_dir_name,
)


class SpikesDetectorAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header
    ) -> Optional[Tuple[List[int], int, float, List[str], float, float, Optional[dict], bool]]:
        groups = self.ensure_non_aux_groups(session_manager, "Spikes detector")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Spikes detector")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, 1e6)
        threshold_spin.setDecimals(6)
        threshold_spin.setSingleStep(0.1)
        threshold_spin.setValue(float(self.state.detector_threshold))
        form.addRow("Threshold (MAD):", threshold_spin)

        detect_upward_checkbox = QCheckBox("Detect upward spikes")
        detect_upward_checkbox.setChecked(bool(self.state.detector_detect_upward_spikes))
        form.addRow("Polarity:", detect_upward_checkbox)

        events_list, before_spin, after_spin = self.build_ignore_events_controls(form, session_manager)
        filter_editor = FilterEditor(form, "Filtering", self.state.detector_filter)

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

        channel_indexes = self.selected_channels(channels_list)
        if not channel_indexes:
            QMessageBox.warning(dialog, "Spikes detector", "Select at least one channel.")
            return None
        group_idx = int(group_combo.currentData())
        ignore_event_names = self.selected_ignore_event_names(events_list)
        ignore_before_ms = float(before_spin.value())
        ignore_after_ms = float(after_spin.value())
        filter_spec = filter_editor.get_filter_spec()
        self.persist_common_selection(
            group_idx=group_idx,
            channels=channel_indexes,
            ignore_event_names=ignore_event_names,
            ignore_before_ms=ignore_before_ms,
            ignore_after_ms=ignore_after_ms,
        )
        self.state.detector_threshold = float(threshold_spin.value())
        self.state.detector_filter = filter_spec
        self.state.detector_detect_upward_spikes = detect_upward_checkbox.isChecked()
        return (
            channel_indexes,
            group_idx,
            float(threshold_spin.value()),
            ignore_event_names,
            ignore_before_ms,
            ignore_after_ms,
            filter_spec,
            detect_upward_checkbox.isChecked(),
        )

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header)
        if params is None:
            return
        (
            channel_indexes,
            _group_idx,
            threshold,
            ignore_event_names,
            ignore_before_ms,
            ignore_after_ms,
            filter_spec,
            detect_upward_spikes,
        ) = params

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        start_sample = 0
        end_sample = sweep_points
        output_number_of_dots = sweep_points
        sample_rate = float(header.sample_rate)
        start_second = 0.0
        end_second = sweep_points / sample_rate

        threshold_dir = Path(add_on_data_dir) / safe_threshold_dir_name(threshold)
        threshold_dir.mkdir(parents=True, exist_ok=True)
        self.state.selected_detector_threshold_dir = str(threshold_dir)

        event_times = self.event_times_by_name_for_window(
            session_manager=session_manager,
            sweep_idx=sweep_idx,
            start_second=start_second,
            end_second=end_second,
        )
        ignore_rules = []
        if ignore_event_names and (ignore_before_ms > 0.0 or ignore_after_ms > 0.0):
            ignore_rules = [
                IgnoreEventsRule(
                    event_names=ignore_event_names,
                    before_ms=ignore_before_ms,
                    after_ms=ignore_after_ms,
                )
            ]
        valid_mask = build_valid_mask(
            n_samples=sweep_points,
            sampling_rate=sample_rate,
            event_times_by_name=event_times,
            ignore_event_rules=ignore_rules,
        )

        selected_filter = filter_from_spec(filter_spec)
        one_filter = [selected_filter] if selected_filter is not None else []

        spikes_by_channel = {}
        total = len(channel_indexes)
        yield {"progress": 0, "message": f"Detecting spikes on sweep {sweep_idx}..."}
        for idx, channel_idx in enumerate(channel_indexes, start=1):
            signal = session_manager.experiment_data.process_single_channel(
                channel_idx=int(channel_idx),
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=1,
                sample_rate=sample_rate,
                filters=one_filter,
                output_number_of_dots=output_number_of_dots,
                transformation_add_ons=[],
            )
            signal = np.asarray(signal, dtype=np.float64)
            mad = float(np.median(np.abs(signal - np.median(signal))))
            threshold_value = threshold * mad / 0.6745 if mad > 0 else threshold
            refractory_samples = max(1, int(round(0.002 * sample_rate)))
            if detect_upward_spikes:
                peaks, _ = find_peaks(
                    signal,
                    height=threshold_value,
                    distance=refractory_samples,
                )
            else:
                peaks, _ = find_peaks(
                    -signal,
                    height=threshold_value,
                    distance=refractory_samples,
                )
            if valid_mask.size == signal.size:
                peaks = peaks[valid_mask[peaks]]
            values = signal[peaks]
            spikes_by_channel[int(channel_idx)] = [
                SpikePoint(
                    time_ms=float((peak * 1000.0) / sample_rate),
                    value=float(value),
                )
                for peak, value in zip(peaks, values)
            ]
            yield {"progress": int((idx / total) * 100), "message": f"Processed channel {idx}/{total}"}

        payload = SpikesPayload(
            threshold=threshold,
            sweep_idx=sweep_idx,
            sample_rate=sample_rate,
            detector_filter=filter_spec,
            ignore_event_names=ignore_event_names,
            ignore_before_ms=ignore_before_ms,
            ignore_after_ms=ignore_after_ms,
            detect_upward_spikes=detect_upward_spikes,
            spikes_by_channel=spikes_by_channel,
        )
        output_path = threshold_dir / f"{sweep_idx}.spikes.json"
        self.save_payload(output_path, payload)
        yield {"progress": 100, "message": f"Saved spikes to {output_path.name}"}
