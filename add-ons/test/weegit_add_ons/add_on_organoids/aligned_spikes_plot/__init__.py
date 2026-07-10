from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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

from weegit_add_ons.add_on_organoids.organoids_common import NIRBaseAddOn, filter_from_spec, safe_file_name


class AlignedSpikesPlotAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[List[int], int, bool, bool, str, Path, float, float, bool, bool, bool, bool, bool, bool, bool]]:
        groups = self.ensure_non_aux_groups(session_manager, "Aligned spikes plot")
        if groups is None:
            return None
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_threshold_dir = self.choose_threshold_dir_dialog("Aligned spikes plot", threshold_dirs)
        if selected_threshold_dir is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Aligned spikes plot")
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

        window_from_spin = QDoubleSpinBox()
        window_from_spin.setRange(0.0, max(0.0, sweep_duration_ms))
        window_from_spin.setDecimals(3)
        window_from_spin.setValue(max(0.0, min(default_from_ms, sweep_duration_ms)))
        form.addRow("Window from (ms):", window_from_spin)

        window_to_spin = QDoubleSpinBox()
        window_to_spin.setRange(0.0, max(0.0, sweep_duration_ms))
        window_to_spin.setDecimals(3)
        window_to_spin.setValue(max(0.0, min(default_to_ms, sweep_duration_ms)))
        form.addRow("Window to (ms):", window_to_spin)

        save_checkbox = QCheckBox("Save images to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        form.addRow("Output:", save_checkbox)
        plot_image_checkbox = QCheckBox("Plot image")
        plot_image_checkbox.setChecked(bool(self.state.aligned_plot_show_image))
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

        x_axis = QCheckBox("x axis is on")
        x_axis.setChecked(bool(self.state.aligned_plot_x_axis))
        form.addRow("Optional:", x_axis)
        y_axis = QCheckBox("y axis is on")
        y_axis.setChecked(bool(self.state.aligned_plot_y_axis))
        form.addRow("", y_axis)
        bg_mesh = QCheckBox("background mesh")
        bg_mesh.setChecked(bool(self.state.aligned_plot_background_mesh))
        form.addRow("", bg_mesh)
        middle_line = QCheckBox("middle line")
        middle_line.setChecked(bool(self.state.aligned_plot_middle_line))
        form.addRow("", middle_line)
        scale_x = QCheckBox("scale bar x axis is on")
        scale_x.setChecked(bool(self.state.aligned_plot_scale_bar_x))
        form.addRow("", scale_x)
        scale_y = QCheckBox("scale bar y axis is on")
        scale_y.setChecked(bool(self.state.aligned_plot_scale_bar_y))
        form.addRow("", scale_y)
        title = QCheckBox("title")
        title.setChecked(bool(self.state.aligned_plot_title))
        form.addRow("", title)

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
            QMessageBox.warning(dialog, "Aligned spikes plot", "Select at least one channel.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Aligned spikes plot", "Select output folder or disable saving.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Aligned spikes plot", "Window 'to' must be greater than 'from'.")
            return None

        self.state.output_folder = folder_edit.text().strip()
        self.state.aligned_plot_x_axis = x_axis.isChecked()
        self.state.aligned_plot_y_axis = y_axis.isChecked()
        self.state.aligned_plot_background_mesh = bg_mesh.isChecked()
        self.state.aligned_plot_middle_line = middle_line.isChecked()
        self.state.aligned_plot_scale_bar_x = scale_x.isChecked()
        self.state.aligned_plot_scale_bar_y = scale_y.isChecked()
        self.state.aligned_plot_title = title.isChecked()
        self.state.aligned_plot_show_image = plot_image_checkbox.isChecked()
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
            channels,
            int(group_combo.currentData()),
            save_checkbox.isChecked(),
            plot_image_checkbox.isChecked(),
            folder_edit.text().strip(),
            selected_threshold_dir,
            window_from_ms,
            window_to_ms,
            x_axis.isChecked(),
            y_axis.isChecked(),
            bg_mesh.isChecked(),
            middle_line.isChecked(),
            scale_x.isChecked(),
            scale_y.isChecked(),
            title.isChecked(),
        )

    @staticmethod
    def _draw_scale_bars(ax, x_values: np.ndarray, y_values: np.ndarray, show_x: bool, show_y: bool) -> None:
        if x_values.size == 0 or y_values.size == 0:
            return
        x_min = float(np.min(x_values))
        x_max = float(np.max(x_values))
        y_min = float(np.min(y_values))
        y_max = float(np.max(y_values))
        x_span = max(1e-12, x_max - x_min)
        y_span = max(1e-12, y_max - y_min)
        x0 = x_min - 0.04 * x_span
        y0 = y_min - 0.04 * y_span
        if show_x:
            bar_x = x_span * 0.25
            ax.plot([x0, x0 + bar_x], [y0, y0], color="black", linewidth=1.2)
            ax.text(x0, y0 + 0.03 * y_span, f"{bar_x:.3g}ms", fontsize=8)
        if show_y:
            bar_y = y_span * 0.5
            ax.plot([x0, x0], [y0, y0 + bar_y], color="black", linewidth=1.2)
            ax.text(x0 + 0.02 * x_span, y0 + bar_y, f"{bar_y:.3g}uV", fontsize=8)

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        (
            channel_indexes,
            _group_idx,
            save_outputs,
            plot_image,
            output_folder,
            threshold_dir,
            window_from_ms,
            window_to_ms,
            x_axis_on,
            y_axis_on,
            background_mesh,
            middle_line_on,
            scale_x_on,
            scale_y_on,
            title_on,
        ) = params

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        payload = self.read_spikes_payload(threshold_dir, sweep_idx)
        if payload is None:
            QMessageBox.warning(None, "Aligned spikes plot", "No spikes file for current sweep in selected threshold folder.")
            return
        detector_filter = filter_from_spec(payload.detector_filter)
        sample_rate = float(header.sample_rate)
        window_start_s = float(window_from_ms) / 1000.0
        window_end_s = float(window_to_ms) / 1000.0
        output_dir = Path(output_folder) if save_outputs else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        window_ms = 4.0
        spike_position_ms = 1.0
        window_samples = max(2, int(round(window_ms * sample_rate / 1000.0)))
        spike_pos_samples = int(round(spike_position_ms * sample_rate / 1000.0))
        start_offset = spike_pos_samples
        end_offset = window_samples - spike_pos_samples

        total = len(channel_indexes)
        yield {"progress": 0, "message": "Generating aligned spikes plots..."}
        for idx, channel_idx in enumerate(channel_indexes, start=1):
            channel_name = self.channel_name(header, int(channel_idx))
            sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
            signal = session_manager.experiment_data.process_single_channel(
                channel_idx=int(channel_idx),
                sweep_idx=sweep_idx,
                start_sample=0,
                end_sample=sweep_points,
                each_point=1,
                sample_rate=sample_rate,
                filters=[detector_filter] if detector_filter is not None else [],
                output_number_of_dots=sweep_points,
                transformation_add_ons=[],
            )
            signal = np.asarray(signal, dtype=np.float64)
            spikes = payload.spikes_by_channel.get(int(channel_idx), [])
            if not spikes:
                yield {"progress": int((idx / total) * 100), "message": f"No spikes for channel {channel_idx}"}
                continue

            waveforms = []
            for spike in spikes:
                spike_time_s = float(spike.time_ms) / 1000.0
                if spike_time_s < window_start_s or spike_time_s > window_end_s:
                    continue
                center = int(round((float(spike.time_ms) / 1000.0) * sample_rate))
                s0 = center - start_offset
                s1 = center + end_offset
                if s0 < 0 or s1 >= signal.size:
                    continue
                waveforms.append(signal[s0:s1])
            if not waveforms:
                yield {"progress": int((idx / total) * 100), "message": f"No valid waveforms for channel {channel_idx}"}
                continue
            waveforms_arr = np.asarray(waveforms, dtype=np.float64)
            time_ms = np.linspace(-spike_position_ms, window_ms - spike_position_ms, waveforms_arr.shape[1])
            mean_waveform = np.mean(waveforms_arr, axis=0)

            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            for waveform in waveforms_arr:
                ax.plot(time_ms, waveform, color="grey", alpha=0.2, linewidth=0.6)
            ax.plot(time_ms, mean_waveform, color="black", linewidth=1.6)
            if middle_line_on:
                ax.axvline(0.0, color="blue", linestyle="--", linewidth=0.8)
            if title_on:
                ax.set_title(f"Aligned spikes {channel_name} (n={len(waveforms_arr)})")
            if x_axis_on:
                ax.set_xlabel("Time (ms)")
            else:
                ax.set_xlabel("")
                ax.set_xticks([])
            if y_axis_on:
                ax.set_ylabel("Amplitude (uV)")
            else:
                ax.set_ylabel("")
                ax.set_yticks([])
            if background_mesh:
                ax.grid(True, alpha=0.25)
            self._draw_scale_bars(ax, time_ms, mean_waveform, scale_x_on, scale_y_on)

            if output_dir is not None:
                fig.savefig(
                    str(output_dir / f"aligned_spikes_{safe_file_name(channel_name)}_sw{sweep_idx}.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
            if plot_image:
                plt.show()
            else:
                plt.close(fig)
            yield {"progress": int((idx / total) * 100), "message": f"Plotted channel {idx}/{total}"}
