from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

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

from weegit_add_ons.add_on_organoids.organoids_common import NIRBaseAddOn, safe_file_name


class RasterPlotAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[List[int], int, bool, bool, str, Path]]:
        groups = self.ensure_non_aux_groups(session_manager, "Raster plot")
        if groups is None:
            return None
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_threshold_dir = self.choose_threshold_dir_dialog("Raster plot", threshold_dirs)
        if selected_threshold_dir is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Raster plot")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        save_checkbox = QCheckBox("Save image to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        form.addRow("Output:", save_checkbox)
        plot_image_checkbox = QCheckBox("Plot image")
        plot_image_checkbox.setChecked(bool(self.state.raster_plot_show_image))
        form.addRow("", plot_image_checkbox)

        folder_edit = QLineEdit(self.state.output_folder or "")
        folder_edit.setEnabled(save_checkbox.isChecked())
        folder_btn = QPushButton("Browse")
        folder_btn.setEnabled(save_checkbox.isChecked())
        folder_row = QHBoxLayout()
        folder_row.addWidget(folder_edit, 1)
        folder_row.addWidget(folder_btn)
        form.addRow("Folder:", folder_row)

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
            QMessageBox.warning(dialog, "Raster plot", "Select at least one channel.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Raster plot", "Select output folder or disable saving.")
            return None
        self.state.output_folder = folder_edit.text().strip()
        self.state.raster_plot_show_image = plot_image_checkbox.isChecked()
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=self.state.ignore_event_names,
            ignore_before_ms=self.state.ignore_before_ms,
            ignore_after_ms=self.state.ignore_after_ms,
        )
        return (
            channels,
            int(group_combo.currentData()),
            save_checkbox.isChecked(),
            plot_image_checkbox.isChecked(),
            folder_edit.text().strip(),
            selected_threshold_dir,
        )

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        channel_indexes, group_idx, save_outputs, plot_image, output_folder, threshold_dir = params

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        payload = self.read_spikes_payload(threshold_dir, sweep_idx)
        if payload is None:
            QMessageBox.warning(None, "Raster plot", "No spikes file for current sweep in selected threshold folder.")
            return

        prev_font_size = plt.rcParams['font.size']
        plt.rcParams.update({'font.size': 14})
        sample_rate = float(header.sample_rate)
        start_sample = int(session_manager.gui_setup.start_point)
        duration_samples = int(round(float(session_manager.gui_setup.duration_ms) * sample_rate / 1000.0))
        end_sample = min(int(header.number_of_points_per_sweep[sweep_idx]), start_sample + max(1, duration_samples))
        start_second = start_sample / sample_rate
        end_second = end_sample / sample_rate

        all_spike_times = []
        all_spike_rows = []
        labels = []
        # Force channel ordering exactly as in selected channel group.
        group = session_manager.gui_setup.channels_groups[int(group_idx)]
        group_order = [int(ch) for ch in (group.channel_indexes or [])]
        group_order.reverse()
        channel_set = set(int(ch) for ch in channel_indexes)
        ordered_channels = [ch for ch in group_order if ch in channel_set]
        for row, channel_idx in enumerate(ordered_channels):
            labels.append(self.channel_name(header, int(channel_idx)))
            for spike in payload.spikes_by_channel.get(int(channel_idx), []):
                t = float(spike.time_ms) / 1000.0
                if start_second <= t <= end_second:
                    all_spike_times.append(t)
                    all_spike_rows.append(row)

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        if all_spike_times:
            ax.plot(np.asarray(all_spike_times), np.asarray(all_spike_rows), "|", color="black", markersize=10)
        ax.set_xlim(start_second, end_second)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        ax.set_yticks(list(range(len(ordered_channels))), labels)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Raster plot | sweep {sweep_idx} | {threshold_dir.name}")

        if save_outputs:
            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            first_name = self.channel_name(header, ordered_channels[0]) if ordered_channels else "channels"
            fig.savefig(
                str(output_dir / f"raster_{safe_file_name(first_name)}_sw{sweep_idx}_{threshold_dir.name}.png"),
                dpi=200,
                bbox_inches="tight",
            )
        if plot_image:
            plt.show()
        else:
            plt.close(fig)
        plt.rcParams.update({'font.size': prev_font_size})
        yield {"progress": 100, "message": "Raster plot generated"}
