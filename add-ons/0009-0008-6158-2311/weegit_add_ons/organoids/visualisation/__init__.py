from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

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
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.signal import spectrogram
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.organoids.common import IgnoreEventsRule, OrganoidsBaseAddOn, build_valid_mask, safe_file_name
from weegit_add_ons.organoids.preprocessing import apply_preprocessing_pipeline, read_pipeline_store
from weegit_add_ons.organoids.spike_utils import (
    avg_spike_csd_classic,
    crosscorrelogram,
    extract_waves,
    spike_time_to_seconds,
    symmetric_autocorrelogram,
)


class VisualisationAddOn(OrganoidsBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    @staticmethod
    def _selected_values(widget: QListWidget) -> List[str]:
        out = []
        for item in widget.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is not None:
                out.append(str(value))
        return out

    @staticmethod
    def _parse_pairs(raw: str, channels: List[int]) -> List[Tuple[int, int]]:
        pairs = []
        for chunk in raw.replace(";", ",").split(","):
            text = chunk.strip()
            if not text or "-" not in text:
                continue
            left, right = text.split("-", 1)
            try:
                a, b = int(left.strip()), int(right.strip())
            except ValueError:
                continue
            if a in channels and b in channels and a != b:
                pairs.append((a, b))
        if not pairs and len(channels) >= 2:
            pairs.append((channels[0], channels[1]))
        return pairs

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path):
        groups = self.ensure_non_aux_groups(session_manager, "Visualisation")
        if groups is None:
            return None
        pipelines = read_pipeline_store(self.pipelines_path(add_on_data_dir))

        dialog = QDialog()
        dialog.setWindowTitle("Visualisation")
        dialog.setMinimumWidth(760)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget()

        common_tab = QWidget()
        common_form = QFormLayout(common_tab)
        group_combo, channels_list = self.build_group_channel_selector(common_form, groups, header)

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        sweep_duration_ms = sweep_points * 1000.0 / sample_rate
        default_from = float(self.state.plots_window_from_ms)
        default_to = float(self.state.plots_window_to_ms)
        if default_to <= default_from:
            default_from = float(session_manager.gui_setup.start_point) * 1000.0 / sample_rate
            default_to = default_from + float(session_manager.gui_setup.duration_ms)

        from_spin = QDoubleSpinBox()
        from_spin.setRange(0.0, sweep_duration_ms)
        from_spin.setDecimals(3)
        from_spin.setValue(max(0.0, min(default_from, sweep_duration_ms)))
        common_form.addRow("Window from (ms):", from_spin)
        to_spin = QDoubleSpinBox()
        to_spin.setRange(0.0, sweep_duration_ms)
        to_spin.setDecimals(3)
        to_spin.setValue(max(0.0, min(default_to, sweep_duration_ms)))
        common_form.addRow("Window to (ms):", to_spin)

        events_list, before_spin, after_spin = self.build_ignore_events_controls(common_form, session_manager)
        pipeline_list = QListWidget()
        pipeline_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for name in sorted(pipelines.keys()):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setSelected(name in {self.state.selected_pipeline_name, "raw"})
            pipeline_list.addItem(item)
        common_form.addRow("Pipelines:", pipeline_list)

        save_checkbox = QCheckBox("Save images to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        common_form.addRow("Output:", save_checkbox)
        show_checkbox = QCheckBox("Show images")
        show_checkbox.setChecked(bool(self.state.visualization_show_image))
        common_form.addRow("", show_checkbox)
        folder_edit = QLineEdit(self.state.output_folder or "")
        browse = QPushButton("Browse")
        folder_row = QHBoxLayout()
        folder_row.addWidget(folder_edit, 1)
        folder_row.addWidget(browse)
        common_form.addRow("Folder:", folder_row)

        def choose_folder() -> None:
            selected = QFileDialog.getExistingDirectory(dialog, "Select output folder", folder_edit.text().strip())
            if selected:
                folder_edit.setText(selected)

        browse.clicked.connect(choose_folder)
        tabs.addTab(common_tab, "Common")

        plot_tab = QWidget()
        plot_form = QFormLayout(plot_tab)
        cb_signal = QCheckBox("Signal")
        cb_signal.setChecked(bool(self.state.visualization_include_signal))
        plot_form.addRow(cb_signal)
        cb_spectrogram = QCheckBox("Spectrogram")
        cb_spectrogram.setChecked(bool(self.state.visualization_include_spectrogram))
        plot_form.addRow(cb_spectrogram)
        cb_csd = QCheckBox("Power/source density (CSD)")
        cb_csd.setChecked(bool(self.state.visualization_include_csd))
        plot_form.addRow(cb_csd)
        cb_aligned = QCheckBox("Aligned spikes")
        cb_aligned.setChecked(bool(self.state.visualization_include_aligned_spikes))
        plot_form.addRow(cb_aligned)
        cb_raster = QCheckBox("Raster plot")
        cb_raster.setChecked(bool(self.state.visualization_include_raster))
        plot_form.addRow(cb_raster)
        cb_auto = QCheckBox("Autocorrelogram")
        cb_auto.setChecked(bool(self.state.visualization_include_autocorrelogram))
        plot_form.addRow(cb_auto)
        cb_cross = QCheckBox("Crosscorrelogram")
        cb_cross.setChecked(bool(self.state.visualization_include_crosscorrelogram))
        plot_form.addRow(cb_cross)
        tabs.addTab(plot_tab, "Plots")

        params_tab = QWidget()
        params_form = QFormLayout(params_tab)
        f_from = QDoubleSpinBox()
        f_from.setRange(0.0, 1_000_000.0)
        f_from.setDecimals(3)
        f_from.setValue(float(self.state.visualization_freq_from_hz))
        params_form.addRow("Frequency from (Hz):", f_from)
        f_to = QDoubleSpinBox()
        f_to.setRange(0.0, 1_000_000.0)
        f_to.setDecimals(3)
        f_to.setValue(float(self.state.visualization_freq_to_hz or 300.0))
        params_form.addRow("Frequency to (Hz):", f_to)
        pairs_edit = QLineEdit()
        pairs_edit.setPlaceholderText("Example: 1-2, 3-4")
        params_form.addRow("Cross pairs:", pairs_edit)
        tabs.addTab(params_tab, "Plot params")

        layout.addWidget(tabs)
        actions = QHBoxLayout()
        cancel = QPushButton("Cancel")
        run = QPushButton("Run")
        actions.addStretch(1)
        actions.addWidget(cancel)
        actions.addWidget(run)
        layout.addLayout(actions)
        cancel.clicked.connect(dialog.reject)
        run.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        channels = self.selected_channels(channels_list)
        if not channels:
            QMessageBox.warning(dialog, "Visualisation", "Select at least one channel.")
            return None
        selected_pipelines = self._selected_values(pipeline_list) or ["raw"]
        if to_spin.value() <= from_spin.value():
            QMessageBox.warning(dialog, "Visualisation", "Window 'to' must be greater than 'from'.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Visualisation", "Select output folder or disable saving.")
            return None
        include = {
            "signal": cb_signal.isChecked(),
            "spectrogram": cb_spectrogram.isChecked(),
            "csd": cb_csd.isChecked(),
            "aligned": cb_aligned.isChecked(),
            "raster": cb_raster.isChecked(),
            "auto": cb_auto.isChecked(),
            "cross": cb_cross.isChecked(),
        }
        ignore_names = self.selected_ignore_event_names(events_list)
        self.persist_common_selection(int(group_combo.currentData()), channels, ignore_names, float(before_spin.value()), float(after_spin.value()))
        self.state.output_folder = folder_edit.text().strip()
        self.state.plots_window_from_ms = float(from_spin.value())
        self.state.plots_window_to_ms = float(to_spin.value())
        self.state.visualization_show_image = show_checkbox.isChecked()
        self.state.visualization_freq_from_hz = float(f_from.value())
        self.state.visualization_freq_to_hz = float(f_to.value())
        self.state.visualization_include_signal = include["signal"]
        self.state.visualization_include_spectrogram = include["spectrogram"]
        self.state.visualization_include_csd = include["csd"]
        self.state.visualization_include_aligned_spikes = include["aligned"]
        self.state.visualization_include_raster = include["raster"]
        self.state.visualization_include_autocorrelogram = include["auto"]
        self.state.visualization_include_crosscorrelogram = include["cross"]
        return {
            "channels": channels,
            "group_idx": int(group_combo.currentData()),
            "pipelines": selected_pipelines,
            "from_ms": float(from_spin.value()),
            "to_ms": float(to_spin.value()),
            "ignore_names": ignore_names,
            "ignore_before": float(before_spin.value()),
            "ignore_after": float(after_spin.value()),
            "save": save_checkbox.isChecked(),
            "show": show_checkbox.isChecked(),
            "folder": folder_edit.text().strip(),
            "include": include,
            "freq_from": float(f_from.value()),
            "freq_to": float(f_to.value()),
            "pairs": pairs_edit.text().strip(),
        }

    def _load_window(self, session_manager, header, cfg):
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        sample_rate = float(header.sample_rate)
        sweep_points = int(header.number_of_points_per_sweep[sweep_idx])
        start = max(0, min(int(round(cfg["from_ms"] / 1000.0 * sample_rate)), sweep_points - 1))
        end = max(start + 1, min(int(round(cfg["to_ms"] / 1000.0 * sample_rate)), sweep_points))
        matrix = self.channel_matrix_from_session(session_manager, cfg["channels"], sweep_idx, start, end, sample_rate)
        times = np.arange(end - start, dtype=np.float64) / sample_rate + start / sample_rate
        event_times = self.event_times_by_name_for_window(session_manager, sweep_idx, start / sample_rate, end / sample_rate)
        rules = []
        if cfg["ignore_names"] and (cfg["ignore_before"] > 0 or cfg["ignore_after"] > 0):
            rules = [IgnoreEventsRule(event_names=cfg["ignore_names"], before_ms=cfg["ignore_before"], after_ms=cfg["ignore_after"])]
        mask = build_valid_mask(end - start, sample_rate, event_times, rules)
        return sweep_idx, sample_rate, start, end, times, matrix, mask

    def _finish_fig(self, fig, output_dir: Path | None, name: str, show: bool) -> None:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_dir / name), dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        cfg = self._ask_parameters(session_manager, header, add_on_data_dir)
        if cfg is None:
            return
        pipelines = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        sweep_idx, sample_rate, start, end, times, raw_matrix, mask = self._load_window(session_manager, header, cfg)
        output_dir = Path(cfg["folder"]) if cfg["save"] else None
        processed_by_pipeline = {}
        for name in cfg["pipelines"]:
            processed = apply_preprocessing_pipeline(raw_matrix, sample_rate, pipelines.get(name))
            if mask.size == processed.shape[1]:
                processed = processed.copy()
                processed[:, ~mask] = 0.0
            processed_by_pipeline[name] = processed

        spikes_dir = None
        spikes_payload = None
        if any(cfg["include"][key] for key in ("aligned", "raster", "auto", "cross", "csd")):
            spikes_dir = self.choose_spikes_dir_dialog("Visualisation", add_on_data_dir)
            if spikes_dir is not None:
                spikes_payload = self.read_spikes_payload(spikes_dir, sweep_idx)

        total_steps = sum(1 for enabled in cfg["include"].values() if enabled)
        done = 0
        yield {"progress": 0, "message": "Generating visualisations..."}

        if cfg["include"]["signal"]:
            for row, channel_idx in enumerate(cfg["channels"]):
                fig, ax = plt.subplots(1, 1, figsize=(14, 4))
                for name, matrix in processed_by_pipeline.items():
                    ax.plot(times, matrix[row], linewidth=0.7, label=name)
                ax.set_title(f"Signal | {self.channel_name(header, channel_idx)} | sweep {sweep_idx}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("uV")
                ax.grid(alpha=0.25)
                ax.legend(loc="upper right")
                self._finish_fig(fig, output_dir, f"signal_{safe_file_name(self.channel_name(header, channel_idx))}_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Signal plots generated"}

        if cfg["include"]["spectrogram"]:
            first_name = cfg["pipelines"][0]
            matrix = processed_by_pipeline[first_name]
            for row, channel_idx in enumerate(cfg["channels"]):
                f, t, power = spectrogram(matrix[row], fs=sample_rate, nperseg=min(1024, max(64, matrix.shape[1] // 8)))
                freq_mask = (f >= cfg["freq_from"]) & (f <= cfg["freq_to"])
                fig, ax = plt.subplots(1, 1, figsize=(12, 5))
                ax.pcolormesh(t + times[0], f[freq_mask], power[freq_mask], shading="auto", cmap="viridis")
                ax.set_title(f"Spectrogram | {self.channel_name(header, channel_idx)} | {first_name}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Frequency (Hz)")
                self._finish_fig(fig, output_dir, f"spectrogram_{safe_file_name(self.channel_name(header, channel_idx))}_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Spectrograms generated"}

        if cfg["include"]["csd"]:
            first_name = cfg["pipelines"][0]
            matrix = processed_by_pipeline[first_name]
            if spikes_payload is not None:
                source_ch = cfg["channels"][0]
                spike_times = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(source_ch), []))
                csd = avg_spike_csd_classic(matrix, spike_times, sample_rate, np.arange(matrix.shape[0], dtype=float) * 50.0)
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                if csd["csd"].size:
                    ax.imshow(csd["csd"], aspect="auto", origin="lower", cmap="coolwarm", extent=[csd["t_ms"][0], csd["t_ms"][-1], 0, matrix.shape[0] - 1])
                ax.set_title(f"Power/source density | {first_name}")
                ax.set_xlabel("Time around spike (ms)")
                ax.set_ylabel("Channel row")
                self._finish_fig(fig, output_dir, f"csd_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "CSD generated"}

        if spikes_payload is not None and cfg["include"]["aligned"]:
            first_name = cfg["pipelines"][0]
            matrix = processed_by_pipeline[first_name]
            for row, channel_idx in enumerate(cfg["channels"]):
                st = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(channel_idx), []))
                st = st[(st >= times[0]) & (st <= times[-1])]
                waves, _centers = extract_waves(matrix[row], st - times[0], sample_rate, pre_ms=2.0, post_ms=2.0)
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                x_ms = np.arange(waves.shape[1], dtype=float) * 1000.0 / sample_rate - 2.0 if waves.size else np.array([])
                for wf in waves:
                    ax.plot(x_ms, wf, color="gray", alpha=0.2, linewidth=0.6)
                if waves.size:
                    ax.plot(x_ms, np.mean(waves, axis=0), color="black", linewidth=1.5)
                ax.axvline(0.0, color="tab:blue", linestyle="--", linewidth=0.8)
                ax.set_title(f"Aligned spikes | {self.channel_name(header, channel_idx)} | n={waves.shape[0]}")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("uV")
                ax.grid(alpha=0.25)
                self._finish_fig(fig, output_dir, f"aligned_{safe_file_name(self.channel_name(header, channel_idx))}_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Aligned spikes generated"}

        if spikes_payload is not None and cfg["include"]["raster"]:
            fig, ax = plt.subplots(1, 1, figsize=(14, 6))
            ordered = list(reversed(cfg["channels"]))
            for row, ch in enumerate(ordered):
                st = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(ch), []))
                st = st[(st >= times[0]) & (st <= times[-1])]
                if st.size:
                    ax.plot(st, np.full(st.size, row), "|", color="black", markersize=10)
            ax.set_yticks(range(len(ordered)), [self.channel_name(header, ch) for ch in ordered])
            ax.set_xlim(times[0], times[-1])
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Channel")
            ax.set_title("Raster plot")
            ax.grid(alpha=0.25)
            self._finish_fig(fig, output_dir, f"raster_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Raster generated"}

        if spikes_payload is not None and cfg["include"]["auto"]:
            for ch in cfg["channels"]:
                st = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(ch), []))
                lags, hist = symmetric_autocorrelogram(st, sample_rate)
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.bar(lags, hist, width=np.median(np.diff(lags)) if lags.size > 1 else 1.0)
                ax.set_title(f"Autocorrelogram | {self.channel_name(header, ch)}")
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Count")
                self._finish_fig(fig, output_dir, f"autocorrelogram_{safe_file_name(self.channel_name(header, ch))}_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Autocorrelograms generated"}

        if spikes_payload is not None and cfg["include"]["cross"]:
            for a, b in self._parse_pairs(cfg["pairs"], cfg["channels"]):
                st_a = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(a), []))
                st_b = spike_time_to_seconds(spikes_payload.spikes_by_channel.get(int(b), []))
                lags, hist = crosscorrelogram(st_a, st_b, sample_rate)
                fig, ax = plt.subplots(1, 1, figsize=(8, 4))
                ax.bar(lags, hist, width=np.median(np.diff(lags)) if lags.size > 1 else 1.0)
                ax.set_title(f"Crosscorrelogram | {self.channel_name(header, a)} vs {self.channel_name(header, b)}")
                ax.set_xlabel("Lag (ms)")
                ax.set_ylabel("Count")
                self._finish_fig(fig, output_dir, f"crosscorrelogram_{a}_{b}_sw{sweep_idx}.png", cfg["show"])
            done += 1
            yield {"progress": int(done / max(1, total_steps) * 100), "message": "Crosscorrelograms generated"}

        yield {"progress": 100, "message": "Visualisation complete"}
