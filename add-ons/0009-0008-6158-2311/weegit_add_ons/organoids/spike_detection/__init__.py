from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
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

from weegit_add_ons.organoids.common import (
    IgnoreEventsRule,
    OrganoidsBaseAddOn,
    SpikePoint,
    SpikesPayload,
    build_valid_mask,
)
from weegit_add_ons.organoids.preprocessing import apply_preprocessing_pipeline, read_pipeline_store
from weegit_add_ons.organoids.spike_utils import (
    detect_spikes_adaptive_mad,
    detect_spikes_mad,
    merge_spikes_global,
    rolling_sigma_mad,
)


def safe_detection_dir_name(pipeline_name: str, threshold: float, adaptive: bool) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pipeline_name.strip()) or "raw"
    mult = f"{float(threshold):.6f}".rstrip("0").rstrip(".").replace(".", "_")
    mode = "adaptive" if adaptive else "global"
    return f"{clean}_{mode}_mad_{mult}"


class SpikeDetectionAddOn(OrganoidsBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[List[int], str, float, bool, bool, bool, float, dict, List[str], float, float]]:
        groups = self.ensure_non_aux_groups(session_manager, "Spike detection")
        if groups is None:
            return None
        pipelines = read_pipeline_store(self.pipelines_path(add_on_data_dir))

        dialog = QDialog()
        dialog.setWindowTitle("Spike detection")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        pipeline_combo = QComboBox()
        for name in sorted(pipelines.keys()):
            pipeline_combo.addItem(name, name)
        idx = pipeline_combo.findData(self.state.selected_pipeline_name)
        pipeline_combo.setCurrentIndex(max(0, idx))
        form.addRow("Preprocessing pipeline:", pipeline_combo)

        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(0.0, 1_000_000.0)
        threshold_spin.setDecimals(6)
        threshold_spin.setSingleStep(0.1)
        threshold_spin.setValue(float(self.state.detector_threshold))
        form.addRow("Threshold multiplier:", threshold_spin)

        adaptive_checkbox = QCheckBox("Use adaptive rolling sigma")
        adaptive_checkbox.setChecked(bool(self.state.detector_adaptive_sigma))
        form.addRow("Sigma:", adaptive_checkbox)

        sigma_window = QDoubleSpinBox()
        sigma_window.setRange(1.0, 60_000.0)
        sigma_window.setDecimals(3)
        sigma_window.setValue(float(self.state.sigma_window_ms))
        form.addRow("Sigma window (ms):", sigma_window)

        sigma_step = QDoubleSpinBox()
        sigma_step.setRange(1.0, 60_000.0)
        sigma_step.setDecimals(3)
        sigma_step.setValue(float(self.state.sigma_step_ms))
        form.addRow("Sigma step (ms):", sigma_step)

        sigma_floor = QDoubleSpinBox()
        sigma_floor.setRange(0.0, 1_000_000.0)
        sigma_floor.setDecimals(6)
        sigma_floor.setValue(float(self.state.sigma_floor_uv))
        form.addRow("Sigma floor (uV):", sigma_floor)

        sigma_smooth = QSpinBox()
        sigma_smooth.setRange(1, 1000)
        sigma_smooth.setValue(int(self.state.sigma_smooth_windows))
        form.addRow("Sigma smooth windows:", sigma_smooth)

        detect_negative = QCheckBox("Detect downward spikes")
        detect_negative.setChecked(bool(self.state.detector_detect_negative))
        form.addRow("Polarity:", detect_negative)
        detect_positive = QCheckBox("Detect upward spikes")
        detect_positive.setChecked(bool(self.state.detector_detect_positive))
        form.addRow("", detect_positive)

        merge_window = QDoubleSpinBox()
        merge_window.setRange(0.0, 1000.0)
        merge_window.setDecimals(3)
        merge_window.setValue(float(self.state.detector_merge_window_ms))
        form.addRow("Merge window (ms):", merge_window)

        events_list, before_spin, after_spin = self.build_ignore_events_controls(form, session_manager)

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
            QMessageBox.warning(dialog, "Spike detection", "Select at least one channel.")
            return None
        if not detect_negative.isChecked() and not detect_positive.isChecked():
            QMessageBox.warning(dialog, "Spike detection", "Select at least one polarity.")
            return None

        ignore_names = self.selected_ignore_event_names(events_list)
        ignore_before = float(before_spin.value())
        ignore_after = float(after_spin.value())
        pipeline_name = str(pipeline_combo.currentData())
        sigma_params = {
            "window_ms": float(sigma_window.value()),
            "step_ms": float(sigma_step.value()),
            "sigma_floor_uv": float(sigma_floor.value()),
            "smooth_windows": int(sigma_smooth.value()),
        }
        self.persist_common_selection(int(group_combo.currentData()), channels, ignore_names, ignore_before, ignore_after)
        self.state.selected_pipeline_name = pipeline_name
        self.state.detector_threshold = float(threshold_spin.value())
        self.state.detector_adaptive_sigma = adaptive_checkbox.isChecked()
        self.state.detector_detect_positive = detect_positive.isChecked()
        self.state.detector_detect_negative = detect_negative.isChecked()
        self.state.detector_merge_window_ms = float(merge_window.value())
        self.state.sigma_window_ms = sigma_params["window_ms"]
        self.state.sigma_step_ms = sigma_params["step_ms"]
        self.state.sigma_floor_uv = sigma_params["sigma_floor_uv"]
        self.state.sigma_smooth_windows = sigma_params["smooth_windows"]
        return (
            channels,
            pipeline_name,
            float(threshold_spin.value()),
            adaptive_checkbox.isChecked(),
            detect_negative.isChecked(),
            detect_positive.isChecked(),
            float(merge_window.value()),
            sigma_params,
            ignore_names,
            ignore_before,
            ignore_after,
        )

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        params = self._ask_parameters(session_manager, header, add_on_data_dir)
        if params is None:
            return
        (
            channel_indexes,
            pipeline_name,
            threshold,
            adaptive_sigma,
            detect_negative,
            detect_positive,
            merge_window_ms,
            sigma_params,
            ignore_event_names,
            ignore_before_ms,
            ignore_after_ms,
        ) = params
        pipelines = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        pipeline = pipelines.get(pipeline_name)

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        matrix = self.channel_matrix_from_session(
            session_manager=session_manager,
            channel_indexes=channel_indexes,
            sweep_idx=sweep_idx,
            start_sample=0,
            end_sample=sweep_points,
            sample_rate=sample_rate,
        )
        processed = apply_preprocessing_pipeline(matrix, sample_rate, pipeline)

        event_times = self.event_times_by_name_for_window(session_manager, sweep_idx, 0.0, sweep_points / sample_rate)
        ignore_rules = []
        if ignore_event_names and (ignore_before_ms > 0.0 or ignore_after_ms > 0.0):
            ignore_rules = [IgnoreEventsRule(event_names=ignore_event_names, before_ms=ignore_before_ms, after_ms=ignore_after_ms)]
        valid_mask = build_valid_mask(sweep_points, sample_rate, event_times, ignore_rules)

        spikes_by_channel = {}
        total = len(channel_indexes)
        yield {"progress": 0, "message": f"Detecting spikes with pipeline '{pipeline_name}'..."}
        for row_idx, channel_idx in enumerate(channel_indexes):
            signal = np.asarray(processed[row_idx], dtype=np.float64)
            sigma_t = None
            if adaptive_sigma:
                sigma_t = rolling_sigma_mad(
                    signal,
                    sample_rate,
                    window_ms=float(sigma_params["window_ms"]),
                    step_ms=float(sigma_params["step_ms"]),
                    sigma_floor_uv=float(sigma_params["sigma_floor_uv"]),
                    smooth_windows=int(sigma_params["smooth_windows"]),
                    mask=valid_mask if valid_mask.size == signal.size else None,
                )
            found = []
            if detect_negative:
                found.extend(
                    detect_spikes_adaptive_mad(signal, sample_rate, threshold, merge_window_ms, False, sigma_t, sigma_params["sigma_floor_uv"])
                    if adaptive_sigma
                    else detect_spikes_mad(signal, sample_rate, threshold, merge_window_ms, False, sigma_params["sigma_floor_uv"])
                )
            if detect_positive:
                found.extend(
                    detect_spikes_adaptive_mad(signal, sample_rate, threshold, merge_window_ms, True, sigma_t, sigma_params["sigma_floor_uv"])
                    if adaptive_sigma
                    else detect_spikes_mad(signal, sample_rate, threshold, merge_window_ms, True, sigma_params["sigma_floor_uv"])
                )
            if valid_mask.size == signal.size:
                found = [sp for sp in found if valid_mask[int(sp.sample_idx)]]
            merged = merge_spikes_global(found, sample_rate, min_distance_ms=merge_window_ms) if detect_negative and detect_positive else sorted(found, key=lambda sp: sp.sample_idx)
            spikes_by_channel[int(channel_idx)] = [
                SpikePoint(
                    sample_idx=int(sp.sample_idx),
                    time_ms=float(sp.time_ms),
                    value=float(sp.value_uv),
                    value_uv=float(sp.value_uv),
                    polarity=str(sp.polarity),
                )
                for sp in merged
            ]
            yield {"progress": int(((row_idx + 1) / total) * 100), "message": f"Processed channel {row_idx + 1}/{total}"}

        out_dir = self.spikes_detection_dir(add_on_data_dir) / safe_detection_dir_name(pipeline_name, threshold, adaptive_sigma)
        self.state.selected_spikes_dir = str(out_dir)
        payload = SpikesPayload(
            preprocessing_pipeline=pipeline_name,
            threshold=float(threshold),
            sweep_idx=sweep_idx,
            sample_rate=sample_rate,
            adaptive_sigma=adaptive_sigma,
            sigma_params=sigma_params,
            detect_positive=detect_positive,
            detect_negative=detect_negative,
            merge_window_ms=merge_window_ms,
            ignore_event_names=ignore_event_names,
            ignore_before_ms=ignore_before_ms,
            ignore_after_ms=ignore_after_ms,
            spikes_by_channel=spikes_by_channel,
        )
        output_path = out_dir / f"{sweep_idx}.spikes.json"
        self.save_spikes_payload(output_path, payload)
        yield {"progress": 100, "message": f"Saved spikes to {output_path.name}"}
