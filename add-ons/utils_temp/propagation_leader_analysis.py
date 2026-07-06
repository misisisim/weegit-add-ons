from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from nir_common import NIRBaseAddOn, NetworkBurstsPayload, safe_file_name


class PropagationLeaderAnalysisAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[Tuple[Path, List[int], bool, bool, str]]:
        nb_dirs = self.list_threshold_dirs_for_module(add_on_data_dir, "network_burst_detector")
        selected_nb_dir = self.choose_threshold_dir_dialog(
            "Propagation & leader analysis",
            nb_dirs,
            selected_dir=self.state.selected_propagation_threshold_dir,
            label="Source network bursts:",
        )
        if selected_nb_dir is None:
            return None
        groups = self.ensure_non_aux_groups(session_manager, "Propagation & leader analysis")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Propagation & leader analysis")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        save_checkbox = QCheckBox("Save outputs to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        form.addRow("Output:", save_checkbox)
        plot_checkbox = QCheckBox("Plot image")
        plot_checkbox.setChecked(bool(self.state.aligned_plot_show_image))
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
            QMessageBox.warning(dialog, "Propagation & leader analysis", "Select at least one channel.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Propagation & leader analysis", "Select output folder or disable saving.")
            return None
        self.state.selected_propagation_threshold_dir = str(selected_nb_dir)
        self.state.output_folder = folder_edit.text().strip()
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=self.state.ignore_event_names,
            ignore_before_ms=self.state.ignore_before_ms,
            ignore_after_ms=self.state.ignore_after_ms,
        )
        return selected_nb_dir, channels, save_checkbox.isChecked(), plot_checkbox.isChecked(), folder_edit.text().strip()

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        selected_nb_dir, channels, save_outputs, plot_image, output_folder = params

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        nb_path = Path(selected_nb_dir) / f"{sweep_idx}.network_bursts.json"
        if not nb_path.exists():
            QMessageBox.warning(None, "Propagation & leader analysis", "No network burst payload for current sweep.")
            return
        nb_payload = NetworkBurstsPayload.model_validate_json(nb_path.read_text(encoding="utf-8"))

        spikes_dir = self.detector_dir_from_any_add_on_dir(Path(add_on_data_dir)) / selected_nb_dir.name
        spikes_payload = self.read_spikes_payload(spikes_dir, sweep_idx)
        if spikes_payload is None:
            QMessageBox.warning(None, "Propagation & leader analysis", "No source spikes payload for network bursts.")
            return

        spike_times = {
            int(ch): np.asarray([float(point.time_ms) for point in spikes_payload.spikes_by_channel.get(int(ch), [])], dtype=float)
            for ch in channels
        }
        for arr in spike_times.values():
            arr.sort()

        leader_counts: Dict[int, int] = {int(ch): 0 for ch in channels}
        latencies_by_channel: Dict[int, List[float]] = {int(ch): [] for ch in channels}
        records = []
        total = max(1, len(nb_payload.network_bursts))
        yield {"progress": 0, "message": "Computing leaders and propagation..."}
        for i, nb in enumerate(nb_payload.network_bursts, start=1):
            first_times = {}
            for ch in channels:
                arr = spike_times[int(ch)]
                if arr.size == 0:
                    continue
                idx = np.searchsorted(arr, float(nb.start_ms), side="left")
                if idx >= arr.size:
                    continue
                t = float(arr[idx])
                if t <= float(nb.end_ms):
                    first_times[int(ch)] = t
            if not first_times:
                continue
            leader_ch = min(first_times.keys(), key=lambda ch: first_times[ch])
            leader_t = float(first_times[leader_ch])
            leader_counts[leader_ch] += 1
            for ch, t in first_times.items():
                latencies_by_channel[int(ch)].append(float(t - leader_t))
            records.append(
                {
                    "start_ms": float(nb.start_ms),
                    "end_ms": float(nb.end_ms),
                    "leader_channel": int(leader_ch),
                    "leader_time_ms": float(leader_t),
                    "first_spike_times_ms": {str(ch): float(t) for ch, t in first_times.items()},
                }
            )
            yield {"progress": int((i / total) * 90), "message": f"Processed network burst {i}/{total}"}

        latency_mean = {
            int(ch): (float(np.mean(vals)) if vals else float("nan")) for ch, vals in latencies_by_channel.items()
        }
        payload = {
            "sweep_idx": sweep_idx,
            "source_network_burst_dir": str(selected_nb_dir),
            "leader_counts": {str(ch): int(v) for ch, v in leader_counts.items()},
            "latency_mean_ms": {str(ch): val for ch, val in latency_mean.items()},
            "records": records,
        }

        canonical_dir = Path(add_on_data_dir) / selected_nb_dir.name
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_json = canonical_dir / f"{sweep_idx}.propagation_leaders.json"
        canonical_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

        output_dir = Path(output_folder) if save_outputs else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            out_json = output_dir / f"propagation_leaders_{selected_nb_dir.name}_sw{sweep_idx}.json"
            out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

        labels = [self.channel_name(header, int(ch)) for ch in channels]
        values = [leader_counts[int(ch)] for ch in channels]
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.bar(range(len(channels)), values, color="#e67e22")
        ax.set_xticks(range(len(channels)), labels, rotation=90)
        ax.set_ylabel("Leader count")
        ax.set_title("Propagation leader channels")
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        if output_dir is not None:
            out_img = output_dir / f"propagation_leaders_{safe_file_name(selected_nb_dir.name)}_sw{sweep_idx}.png"
            fig.savefig(str(out_img), dpi=200, bbox_inches="tight")
        if plot_image:
            plt.show()
        else:
            plt.close(fig)
        yield {"progress": 100, "message": "Propagation & leader analysis complete"}
