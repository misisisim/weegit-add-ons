from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import Qt
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
    QVBoxLayout, QScrollArea, QWidget,
)
from weegit.core.add_ons.base import BaseAddOn

from nir_common import (
    FilterEditor,
    IgnoreEventsRule,
    NIRBaseAddOn,
    build_valid_mask,
    filter_from_spec,
    safe_file_name,
)


class SignalPlotAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header
    ) -> Optional[
        Tuple[
            List[int],
            int,
            bool,
            bool,
            str,
            float,
            float,
            List[str],
            float,
            float,
            bool,
            bool,
            bool,
            Optional[dict],
            Optional[dict],
            bool,
            bool,
            bool,
            bool,
            bool,
        ]
    ]:
        groups = self.ensure_non_aux_groups(session_manager, "Signal plot")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Signal plot")
        dialog.setFixedHeight(600)
        layout = QVBoxLayout(dialog)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_widget = QWidget()
        form = QFormLayout(scroll_widget)  # Attach form layout to the container

        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)
        events_list, before_spin, after_spin = self.build_ignore_events_controls(form, session_manager)

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
        plot_image_checkbox.setChecked(bool(self.state.signal_plot_show_image))
        form.addRow("", plot_image_checkbox)

        folder_edit = QLineEdit()
        folder_edit.setText(self.state.output_folder or "")
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

        raw_checkbox = QCheckBox("Raw signal")
        raw_checkbox.setChecked(bool(self.state.signal_plot_raw_enabled))
        form.addRow("Plot raw:", raw_checkbox)

        ap_checkbox = QCheckBox("Action potentials")
        ap_checkbox.setChecked(bool(self.state.signal_plot_ap_enabled))
        form.addRow("Plot AP:", ap_checkbox)
        ap_filter_editor = FilterEditor(form, "AP filter", self.state.action_potential_filter)

        lfp_checkbox = QCheckBox("Local field potentials")
        lfp_checkbox.setChecked(bool(self.state.signal_plot_lfp_enabled))
        form.addRow("Plot LFP:", lfp_checkbox)
        lfp_filter_editor = FilterEditor(form, "LFP filter", self.state.local_field_filter)

        x_axis_checkbox = QCheckBox("x axis is on")
        x_axis_checkbox.setChecked(bool(self.state.signal_plot_x_axis))
        form.addRow("Optional:", x_axis_checkbox)

        y_axis_checkbox = QCheckBox("y axis is on")
        y_axis_checkbox.setChecked(bool(self.state.signal_plot_y_axis))
        form.addRow("", y_axis_checkbox)

        scalebar_x_checkbox = QCheckBox("scale bar x axis is on")
        scalebar_x_checkbox.setChecked(bool(self.state.signal_plot_scale_bar_x))
        form.addRow("", scalebar_x_checkbox)

        scalebar_y_checkbox = QCheckBox("scale bar y axis is on")
        scalebar_y_checkbox.setChecked(bool(self.state.signal_plot_scale_bar_y))
        form.addRow("", scalebar_y_checkbox)

        title_checkbox = QCheckBox("title")
        title_checkbox.setChecked(bool(self.state.signal_plot_title))
        form.addRow("", title_checkbox)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)  # Add scroll area (takes most space)

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
            QMessageBox.warning(dialog, "Signal plot", "Select at least one channel.")
            return None
        if not raw_checkbox.isChecked() and not ap_checkbox.isChecked() and not lfp_checkbox.isChecked():
            QMessageBox.warning(dialog, "Signal plot", "Enable at least one signal type.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Signal plot", "Select output folder or disable saving.")
            return None
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Signal plot", "Window 'to' must be greater than 'from'.")
            return None

        ignore_names = self.selected_ignore_event_names(events_list)
        ignore_before = float(before_spin.value())
        ignore_after = float(after_spin.value())
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=ignore_names,
            ignore_before_ms=ignore_before,
            ignore_after_ms=ignore_after,
        )
        self.state.output_folder = folder_edit.text().strip()
        self.state.signal_plot_raw_enabled = raw_checkbox.isChecked()
        self.state.signal_plot_ap_enabled = ap_checkbox.isChecked()
        self.state.signal_plot_lfp_enabled = lfp_checkbox.isChecked()
        self.state.action_potential_filter = ap_filter_editor.get_filter_spec()
        self.state.local_field_filter = lfp_filter_editor.get_filter_spec()
        self.state.signal_plot_x_axis = x_axis_checkbox.isChecked()
        self.state.signal_plot_y_axis = y_axis_checkbox.isChecked()
        self.state.signal_plot_scale_bar_x = scalebar_x_checkbox.isChecked()
        self.state.signal_plot_scale_bar_y = scalebar_y_checkbox.isChecked()
        self.state.signal_plot_title = title_checkbox.isChecked()
        self.state.signal_plot_show_image = plot_image_checkbox.isChecked()
        self.state.plots_window_from_ms = window_from_ms
        self.state.plots_window_to_ms = window_to_ms
        return (
            channels,
            int(group_combo.currentData()),
            save_checkbox.isChecked(),
            plot_image_checkbox.isChecked(),
            folder_edit.text().strip(),
            window_from_ms,
            window_to_ms,
            ignore_names,
            ignore_before,
            ignore_after,
            raw_checkbox.isChecked(),
            ap_checkbox.isChecked(),
            lfp_checkbox.isChecked(),
            ap_filter_editor.get_filter_spec(),
            lfp_filter_editor.get_filter_spec(),
            x_axis_checkbox.isChecked(),
            y_axis_checkbox.isChecked(),
            scalebar_x_checkbox.isChecked(),
            scalebar_y_checkbox.isChecked(),
            title_checkbox.isChecked(),
        )

    @staticmethod
    def _draw_scale_bars(
        ax,
        times: np.ndarray,
        data: np.ndarray,
        show_x_bar: bool,
        show_y_bar: bool,
    ) -> None:
        if times.size == 0 or data.size == 0:
            return
        x_span = float(times[-1] - times[0])
        y_min = float(np.nanmin(data))
        y_max = float(np.nanmax(data))
        y_span = max(1e-12, y_max - y_min)
        x0 = float(times[0]) + 0.05 * x_span
        y0 = y_min + 0.10 * y_span

        if show_x_bar:
            bar_x = max(0.001, x_span * 0.1)
            ax.plot([x0, x0 + bar_x], [y0, y0], color="black", linewidth=1.2)
            ax.text(x0, y0 + 0.03 * y_span, f"{bar_x:.3g}s", fontsize=8)
        if show_y_bar:
            bar_y = max(1e-9, y_span * 0.1)
            ax.plot([x0, x0], [y0, y0 + bar_y], color="black", linewidth=1.2)
            ax.text(x0 + 0.01 * x_span, y0 + bar_y, f"{bar_y:.3g}uV", fontsize=8)

    @staticmethod
    def _y_scale_text(data: np.ndarray) -> str:
        if data.size == 0:
            return "Y scale: n/a"
        y_min = float(np.nanmin(data))
        y_max = float(np.nanmax(data))
        span = float(y_max - y_min)
        return f"Y scale: {span:.3g}uV ({y_min:.3g}..{y_max:.3g})"

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header)
        if params is None:
            return
        (
            channel_indexes,
            _group_idx,
            save_outputs,
            plot_image,
            output_folder,
            window_from_ms,
            window_to_ms,
            ignore_event_names,
            ignore_before_ms,
            ignore_after_ms,
            plot_raw,
            plot_ap,
            plot_lfp,
            ap_filter_spec,
            lfp_filter_spec,
            x_axis_on,
            y_axis_on,
            x_scalebar_on,
            y_scalebar_on,
            title_on,
        ) = params

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        start_sample = int(round((window_from_ms / 1000.0) * sample_rate))
        end_sample = int(round((window_to_ms / 1000.0) * sample_rate))
        start_sample = max(0, min(start_sample, sweep_points - 1))
        end_sample = max(start_sample + 1, min(end_sample, sweep_points))
        n_samples = max(1, end_sample - start_sample)
        start_second = start_sample / sample_rate
        end_second = end_sample / sample_rate

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
            n_samples=n_samples,
            sampling_rate=sample_rate,
            event_times_by_name=event_times,
            ignore_event_rules=ignore_rules,
        )
        local_time = np.arange(n_samples, dtype=float) / sample_rate + start_second

        ap_filter = filter_from_spec(ap_filter_spec)
        lfp_filter = filter_from_spec(lfp_filter_spec)
        output_dir = Path(output_folder) if save_outputs else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        total = len(channel_indexes)
        yield {"progress": 0, "message": "Generating signal plots..."}
        for idx, channel_idx in enumerate(channel_indexes, start=1):
            channel_name = self.channel_name(header, int(channel_idx))
            raw = session_manager.experiment_data.process_single_channel(
                channel_idx=int(channel_idx),
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=1,
                sample_rate=sample_rate,
                filters=[],
                output_number_of_dots=n_samples,
                transformation_add_ons=[],
            )
            raw = np.asarray(raw, dtype=np.float64)

            traces: List[Tuple[str, np.ndarray, str]] = []
            if plot_raw:
                traces.append(("Raw", raw.copy(), "#111111"))
            if plot_ap:
                filtered = session_manager.experiment_data.process_single_channel(
                    channel_idx=int(channel_idx),
                    sweep_idx=sweep_idx,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    each_point=1,
                    sample_rate=sample_rate,
                    filters=[ap_filter] if ap_filter is not None else [],
                    output_number_of_dots=n_samples,
                    transformation_add_ons=[],
                )
                traces.append(("AP", np.asarray(filtered, dtype=np.float64), "#1b4dff"))
            if plot_lfp:
                filtered = session_manager.experiment_data.process_single_channel(
                    channel_idx=int(channel_idx),
                    sweep_idx=sweep_idx,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    each_point=1,
                    sample_rate=sample_rate,
                    filters=[lfp_filter] if lfp_filter is not None else [],
                    output_number_of_dots=n_samples,
                    transformation_add_ons=[],
                )
                traces.append(("LFP", np.asarray(filtered, dtype=np.float64), "#0d8f4a"))

            # Keep timeline continuous and zero-fill ignored windows.
            for trace_idx, (_label, signal, _color) in enumerate(traces):
                if valid_mask.size == signal.size:
                    traces[trace_idx] = (traces[trace_idx][0], signal.copy(), traces[trace_idx][2])
                    traces[trace_idx][1][~valid_mask] = 0.0

            n_rows = 1 + len(traces)  # overlay + each individual trace
            fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(4, 2.8 * n_rows)), sharex=True)
            if n_rows == 1:
                axes = [axes]

            overlay_ax = axes[0]
            all_values = []
            for label, signal, color in traces:
                all_values.append(signal)
                overlay_ax.plot(local_time, signal, linewidth=0.7, label=label, color=color)
            overlay_ax.set_xlim(start_second, end_second)
            if title_on:
                overlay_ax.set_title(f"Channel {channel_name} | sweep {sweep_idx}")
            if y_axis_on:
                overlay_ax.set_ylabel("Amplitude (uV)")
            else:
                overlay_ax.set_ylabel("")
                overlay_ax.set_yticks([])
            if x_axis_on or y_axis_on:
                overlay_ax.grid(True, alpha=0.25)
            if traces:
                overlay_ax.legend(loc="upper right")
            if all_values:
                merged = np.concatenate(all_values)
                self._draw_scale_bars(overlay_ax, local_time, merged, x_scalebar_on, y_scalebar_on)

            for row_idx, (label, signal, color) in enumerate(traces, start=1):
                ax = axes[row_idx]
                ax.plot(local_time, signal, linewidth=0.7, color=color)
                ax.set_xlim(start_second, end_second)
                if y_axis_on:
                    ax.set_ylabel("Amplitude (uV)")
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])
                if x_axis_on or y_axis_on:
                    ax.grid(True, alpha=0.25)
                ax.set_title(f"{label} | {self._y_scale_text(signal)}")

            if x_axis_on:
                axes[-1].set_xlabel("Time (s)")
            else:
                axes[-1].set_xlabel("")
                axes[-1].set_xticks([])

            if output_dir is not None:
                out_name = safe_file_name(channel_name)
                fig.savefig(str(output_dir / f"signal_{out_name}_sw{sweep_idx}.png"), dpi=200, bbox_inches="tight")
            if plot_image:
                plt.show()
            else:
                plt.close(fig)
            yield {"progress": int((idx / total) * 100), "message": f"Plotted channel {idx}/{total}"}
