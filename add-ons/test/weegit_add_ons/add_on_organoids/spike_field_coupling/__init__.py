from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.add_on_organoids.organoids_common import FilterEditor, NIRBaseAddOn, filter_from_spec, safe_file_name


class SpikeFieldCouplingAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[Path, List[int], bool, bool, str, float, float, Optional[dict]]]:
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_source_dir = self.choose_threshold_dir_dialog(
            "Spike-field coupling",
            threshold_dirs,
            selected_dir=self.state.selected_sfc_threshold_dir,
            label="Source spikes:",
        )
        if selected_source_dir is None:
            return None
        groups = self.ensure_non_aux_groups(session_manager, "Spike-field coupling")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Spike-field coupling")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        sample_rate = float(header.sample_rate)
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        sweep_duration_ms = (sweep_points / sample_rate) * 1000.0
        default_from_ms = float(self.state.plots_window_from_ms)
        default_to_ms = float(self.state.plots_window_to_ms)
        if default_to_ms <= default_from_ms:
            default_from_ms = float(session_manager.gui_setup.start_point) * 1000.0 / sample_rate
            default_to_ms = default_from_ms + float(session_manager.gui_setup.duration_ms)

        from_spin = QDoubleSpinBox()
        from_spin.setRange(0.0, sweep_duration_ms)
        from_spin.setDecimals(3)
        from_spin.setValue(max(0.0, min(default_from_ms, sweep_duration_ms)))
        form.addRow("Window from (ms):", from_spin)

        to_spin = QDoubleSpinBox()
        to_spin.setRange(0.0, sweep_duration_ms)
        to_spin.setDecimals(3)
        to_spin.setValue(max(0.0, min(default_to_ms, sweep_duration_ms)))
        form.addRow("Window to (ms):", to_spin)

        save_checkbox = QCheckBox("Save outputs to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        form.addRow("Output:", save_checkbox)
        plot_checkbox = QCheckBox("Plot image")
        plot_checkbox.setChecked(bool(self.state.spectrogram_show_image))
        form.addRow("", plot_checkbox)

        folder_edit = QLineEdit(self.state.output_folder or "")
        folder_edit.setEnabled(save_checkbox.isChecked())
        folder_btn = QPushButton("Browse")
        folder_btn.setEnabled(save_checkbox.isChecked())
        row = QHBoxLayout()
        row.addWidget(folder_edit, 1)
        row.addWidget(folder_btn)
        form.addRow("Folder:", row)

        def choose_folder() -> None:
            selected = QFileDialog.getExistingDirectory(dialog, "Select output folder", folder_edit.text().strip())
            if selected:
                folder_edit.setText(selected)

        def on_save_changed(_state: int) -> None:
            enabled = save_checkbox.isChecked()
            folder_edit.setEnabled(enabled)
            folder_btn.setEnabled(enabled)

        folder_btn.clicked.connect(choose_folder)
        save_checkbox.stateChanged.connect(on_save_changed)

        filter_editor = FilterEditor(form, "LFP filter", self.state.local_field_filter)

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
            QMessageBox.warning(dialog, "Spike-field coupling", "Select at least one channel.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Spike-field coupling", "Select output folder or disable saving.")
            return None
        window_from_ms = float(from_spin.value())
        window_to_ms = float(to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Spike-field coupling", "Window 'to' must be greater than 'from'.")
            return None

        self.state.selected_sfc_threshold_dir = str(selected_source_dir)
        self.state.output_folder = folder_edit.text().strip()
        self.state.local_field_filter = filter_editor.get_filter_spec()
        self.state.plots_window_from_ms = window_from_ms
        self.state.plots_window_to_ms = window_to_ms
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
            save_checkbox.isChecked(),
            plot_checkbox.isChecked(),
            folder_edit.text().strip(),
            window_from_ms,
            window_to_ms,
            filter_editor.get_filter_spec(),
        )

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        source_threshold_dir, channels, save_outputs, plot_image, output_folder, window_from_ms, window_to_ms, filter_spec = params
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        spikes_payload = self.read_spikes_payload(source_threshold_dir, sweep_idx)
        if spikes_payload is None:
            QMessageBox.warning(None, "Spike-field coupling", "No spikes payload for current sweep.")
            return

        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        start_sample = int(round((window_from_ms / 1000.0) * sample_rate))
        end_sample = int(round((window_to_ms / 1000.0) * sample_rate))
        start_sample = max(0, min(start_sample, sweep_points - 1))
        end_sample = max(start_sample + 1, min(end_sample, sweep_points))
        n_samples = end_sample - start_sample

        flt = filter_from_spec(filter_spec)
        coupling = {}
        total = len(channels)
        yield {"progress": 0, "message": "Computing spike-field coupling..."}
        for i, ch in enumerate(channels, start=1):
            signal = session_manager.experiment_data.process_single_channel(
                channel_idx=int(ch),
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=1,
                sample_rate=sample_rate,
                filters=[flt] if flt is not None else [],
                output_number_of_dots=n_samples,
                transformation_add_ons=[],
            )
            signal = np.asarray(signal, dtype=np.float64)
            analytic = hilbert(signal)
            phase = np.angle(analytic)
            spike_times_ms = np.asarray(
                [float(point.time_ms) for point in spikes_payload.spikes_by_channel.get(int(ch), [])],
                dtype=float,
            )
            mask = (spike_times_ms >= window_from_ms) & (spike_times_ms <= window_to_ms)
            selected_spikes_ms = spike_times_ms[mask]
            sample_idx = np.round((selected_spikes_ms / 1000.0) * sample_rate).astype(int) - start_sample
            sample_idx = sample_idx[(0 <= sample_idx) & (sample_idx < phase.size)]
            if sample_idx.size == 0:
                plv = 0.0
                pref_phase = 0.0
            else:
                vectors = np.exp(1j * phase[sample_idx])
                mean_vec = np.mean(vectors)
                plv = float(np.abs(mean_vec))
                pref_phase = float(np.angle(mean_vec))
            coupling[int(ch)] = {
                "channel_name": self.channel_name(header, int(ch)),
                "plv": plv,
                "preferred_phase_rad": pref_phase,
                "n_spikes_used": int(sample_idx.size),
            }
            yield {"progress": int((i / total) * 90), "message": f"Processed channel {i}/{total}"}

        output_dir = Path(output_folder) if save_outputs else None
        canonical_dir = Path(add_on_data_dir) / source_threshold_dir.name
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_json = canonical_dir / f"{sweep_idx}.sfc.json"
        canonical_json.write_text(
            json.dumps({"sweep_idx": sweep_idx, "source_threshold_dir": str(source_threshold_dir), "channels": coupling},
                       ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_json = output_dir / f"sfc_{source_threshold_dir.name}_sw{sweep_idx}.json"
            out_json.write_text(json.dumps({"sweep_idx": sweep_idx, "channels": coupling}, ensure_ascii=True, indent=2), encoding="utf-8")

        labels = [coupling[int(ch)]["channel_name"] for ch in channels]
        plv_values = [float(coupling[int(ch)]["plv"]) for ch in channels]
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.bar(range(len(channels)), plv_values, color="#3a6ff7")
        ax.set_xticks(range(len(channels)), labels, rotation=90)
        ax.set_ylabel("PLV")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Spike-field coupling (PLV)")
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        if output_dir is not None:
            out_img = output_dir / f"sfc_{safe_file_name(source_threshold_dir.name)}_sw{sweep_idx}.png"
            fig.savefig(str(out_img), dpi=200, bbox_inches="tight")
        if plot_image:
            plt.show()
        else:
            plt.close(fig)
        yield {"progress": 100, "message": "Spike-field coupling complete"}
