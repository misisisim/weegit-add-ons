from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.organoids.common import OrganoidsBaseAddOn
from weegit_add_ons.organoids.preprocessing import read_pipeline_store


class JupyterGeneratorAddOn(OrganoidsBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    @staticmethod
    def _cell(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}

    @staticmethod
    def _markdown(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    @staticmethod
    def _selected_blocks(blocks_list: QListWidget) -> List[str]:
        out = []
        for item in blocks_list.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is not None:
                out.append(str(value))
        return out

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[dict]:
        groups = self.ensure_non_aux_groups(session_manager, "Jupyter generator")
        if groups is None:
            return None
        pipelines = read_pipeline_store(self.pipelines_path(add_on_data_dir))

        dialog = QDialog()
        dialog.setWindowTitle("Jupyter generator")
        dialog.setMinimumWidth(640)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        pipeline_combo = QComboBox()
        for name in sorted(pipelines.keys()):
            pipeline_combo.addItem(name, name)
        idx = pipeline_combo.findData(self.state.selected_pipeline_name)
        pipeline_combo.setCurrentIndex(max(0, idx))
        form.addRow("Preprocessing pipeline:", pipeline_combo)

        blocks = {
            "signal": "Signal",
            "spectrogram": "Spectrogram",
            "csd": "Power/source density",
            "aligned": "Aligned spikes",
            "raster": "Raster plot",
            "auto": "Autocorrelogram",
            "cross": "Crosscorrelogram",
        }
        blocks_list = QListWidget()
        blocks_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        default_selected = {
            "signal": self.state.notebook_include_signal,
            "spectrogram": self.state.notebook_include_spectrogram,
            "csd": self.state.notebook_include_csd,
            "aligned": self.state.notebook_include_aligned_spikes,
            "raster": self.state.notebook_include_raster,
            "auto": self.state.notebook_include_autocorrelogram,
            "cross": self.state.notebook_include_crosscorrelogram,
        }
        for key, label in blocks.items():
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setSelected(bool(default_selected.get(key, False)))
            blocks_list.addItem(item)
        form.addRow("Notebook blocks:", blocks_list)

        include_spikes = QCheckBox("Include selected spike_detection payload")
        include_spikes.setChecked(True)
        form.addRow("Spikes:", include_spikes)

        default_name = f"organoids_analysis_sweep_{int(session_manager.gui_setup.current_sweep_idx)}.ipynb"
        path_edit = QLineEdit(str(Path(self.state.output_folder) / default_name) if self.state.output_folder else default_name)
        browse = QPushButton("Browse")
        row = QHBoxLayout()
        row.addWidget(path_edit, 1)
        row.addWidget(browse)
        form.addRow("Notebook path:", row)

        def choose_path() -> None:
            selected, _ = QFileDialog.getSaveFileName(dialog, "Select notebook path", path_edit.text().strip() or default_name, "Jupyter Notebook (*.ipynb)")
            if selected:
                if not selected.endswith(".ipynb"):
                    selected += ".ipynb"
                path_edit.setText(selected)

        browse.clicked.connect(choose_path)
        layout.addLayout(form)
        actions = QHBoxLayout()
        cancel = QPushButton("Cancel")
        generate = QPushButton("Generate")
        actions.addStretch(1)
        actions.addWidget(cancel)
        actions.addWidget(generate)
        layout.addLayout(actions)
        cancel.clicked.connect(dialog.reject)
        generate.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        channels = self.selected_channels(channels_list)
        if not channels:
            QMessageBox.warning(dialog, "Jupyter generator", "Select at least one channel.")
            return None
        blocks_selected = self._selected_blocks(blocks_list)
        if not blocks_selected:
            QMessageBox.warning(dialog, "Jupyter generator", "Select at least one notebook block.")
            return None
        notebook_path = path_edit.text().strip()
        if not notebook_path:
            QMessageBox.warning(dialog, "Jupyter generator", "Select output path.")
            return None
        selected_spikes_dir = None
        if include_spikes.isChecked():
            selected_spikes_dir = self.choose_spikes_dir_dialog("Jupyter generator", add_on_data_dir)
        self.persist_common_selection(int(group_combo.currentData()), channels, self.state.ignore_event_names, self.state.ignore_before_ms, self.state.ignore_after_ms)
        self.state.selected_pipeline_name = str(pipeline_combo.currentData())
        self.state.output_folder = str(Path(notebook_path).parent)
        return {
            "channels": channels,
            "pipeline": str(pipeline_combo.currentData()),
            "blocks": blocks_selected,
            "notebook_path": notebook_path,
            "spikes_dir": str(selected_spikes_dir) if selected_spikes_dir else "",
        }

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        add_on_data_dir = Path(add_on_data_dir)
        cfg = self._ask_parameters(session_manager, header, add_on_data_dir)
        if cfg is None:
            return
        notebook_path = Path(cfg["notebook_path"])
        experiment_folder = Path(session_manager.weegit_experiment_folder)
        session_filename = getattr(session_manager.user_session, "session_filename", "")
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        pipelines_path = self.pipelines_path(add_on_data_dir)
        spikes_path = Path(cfg["spikes_dir"]) / f"{sweep_idx}.spikes.json" if cfg["spikes_dir"] else Path("")
        channel_name_map = {int(ch): self.channel_name(header, int(ch)) for ch in cfg["channels"]}

        cells = [
            self._markdown("# Organoids electrophysiology analysis\nGenerated by Weegit organoids add-on.\n"),
            self._cell(
                "from pathlib import Path\n"
                "import json\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "from scipy.signal import spectrogram\n"
                "from weegit.core.weegit_session import WeegitSessionManager\n"
                "from weegit_add_ons.organoids.preprocessing import read_pipeline_store, apply_preprocessing_pipeline\n"
                "from weegit_add_ons.organoids.spike_utils import extract_waves, spike_time_to_seconds, symmetric_autocorrelogram, crosscorrelogram, avg_spike_csd_classic\n"
            ),
            self._cell(
                f"weegit_folder = Path(r'''{experiment_folder}''')\n"
                f"session_filename = r'''{session_filename}'''\n"
                f"sweep_idx = {sweep_idx}\n"
                f"channel_indexes = {cfg['channels']}\n"
                f"channel_name_by_idx = {channel_name_map!r}\n"
                f"pipeline_name = {cfg['pipeline']!r}\n"
                f"pipelines_path = Path(r'''{pipelines_path}''')\n"
                f"spikes_path = Path(r'''{spikes_path}''')\n"
            ),
            self._cell(
                "manager = WeegitSessionManager()\n"
                "manager.init_from_folder(weegit_folder)\n"
                "if session_filename:\n"
                "    manager.switch_sessions(session_filename)\n"
                "header = manager.experiment_data.header\n"
                "sample_rate = float(header.sample_rate)\n"
                "sweep_points = int(header.number_of_points_per_sweep[sweep_idx])\n"
            ),
            self._cell(
                "rows = []\n"
                "for ch in channel_indexes:\n"
                "    signal = manager.experiment_data.process_single_channel(\n"
                "        channel_idx=int(ch), sweep_idx=sweep_idx, start_sample=0, end_sample=sweep_points,\n"
                "        each_point=1, sample_rate=sample_rate, filters=[], output_number_of_dots=sweep_points,\n"
                "        transformation_add_ons=[],\n"
                "    )\n"
                "    rows.append(np.asarray(signal, dtype=np.float64))\n"
                "raw_matrix = np.vstack(rows)\n"
                "pipelines = read_pipeline_store(pipelines_path)\n"
                "pipeline = pipelines.get(pipeline_name)\n"
                "processed = apply_preprocessing_pipeline(raw_matrix, sample_rate, pipeline)\n"
                "times = np.arange(processed.shape[1], dtype=float) / sample_rate\n"
                "spikes_payload = json.loads(spikes_path.read_text(encoding='utf-8')) if spikes_path.exists() else {'spikes_by_channel': {}}\n"
                "spikes_by_channel = {int(k): v for k, v in spikes_payload.get('spikes_by_channel', {}).items()}\n"
            ),
        ]

        if "signal" in cfg["blocks"]:
            cells.append(self._cell(
                "for row, ch in enumerate(channel_indexes):\n"
                "    plt.figure(figsize=(14, 3))\n"
                "    plt.plot(times, processed[row], linewidth=0.7)\n"
                "    plt.title(f\"Signal | {channel_name_by_idx.get(int(ch), ch)} | {pipeline_name}\")\n"
                "    plt.xlabel('Time (s)'); plt.ylabel('uV'); plt.grid(alpha=0.25); plt.show()\n"
            ))
        if "spectrogram" in cfg["blocks"]:
            cells.append(self._cell(
                "for row, ch in enumerate(channel_indexes):\n"
                "    f, t, p = spectrogram(processed[row], fs=sample_rate, nperseg=min(1024, max(64, processed.shape[1] // 8)))\n"
                "    plt.figure(figsize=(12, 4))\n"
                "    plt.pcolormesh(t, f, p, shading='auto', cmap='viridis')\n"
                "    plt.title(f\"Spectrogram | {channel_name_by_idx.get(int(ch), ch)}\")\n"
                "    plt.xlabel('Time (s)'); plt.ylabel('Hz'); plt.colorbar(); plt.show()\n"
            ))
        if "raster" in cfg["blocks"]:
            cells.append(self._cell(
                "plt.figure(figsize=(14, 5))\n"
                "for row, ch in enumerate(reversed(channel_indexes)):\n"
                "    st = spike_time_to_seconds(spikes_by_channel.get(int(ch), []))\n"
                "    if st.size:\n"
                "        plt.plot(st, np.full(st.size, row), '|', color='black', markersize=10)\n"
                "plt.yticks(range(len(channel_indexes)), [channel_name_by_idx.get(int(ch), ch) for ch in reversed(channel_indexes)])\n"
                "plt.xlabel('Time (s)'); plt.ylabel('Channel'); plt.title('Raster plot'); plt.grid(alpha=0.25); plt.show()\n"
            ))
        if "aligned" in cfg["blocks"]:
            cells.append(self._cell(
                "for row, ch in enumerate(channel_indexes):\n"
                "    st = spike_time_to_seconds(spikes_by_channel.get(int(ch), []))\n"
                "    waves, _ = extract_waves(processed[row], st, sample_rate, pre_ms=2.0, post_ms=2.0)\n"
                "    plt.figure(figsize=(8, 4))\n"
                "    if waves.size:\n"
                "        x_ms = np.arange(waves.shape[1]) * 1000.0 / sample_rate - 2.0\n"
                "        for wave in waves:\n"
                "            plt.plot(x_ms, wave, color='gray', alpha=0.2, linewidth=0.6)\n"
                "        plt.plot(x_ms, waves.mean(axis=0), color='black', linewidth=1.5)\n"
                "    plt.axvline(0, color='tab:blue', linestyle='--')\n"
                "    plt.title(f\"Aligned spikes | {channel_name_by_idx.get(int(ch), ch)} | n={waves.shape[0]}\")\n"
                "    plt.xlabel('Time (ms)'); plt.ylabel('uV'); plt.grid(alpha=0.25); plt.show()\n"
            ))
        if "auto" in cfg["blocks"]:
            cells.append(self._cell(
                "for ch in channel_indexes:\n"
                "    st = spike_time_to_seconds(spikes_by_channel.get(int(ch), []))\n"
                "    lags, hist = symmetric_autocorrelogram(st, sample_rate)\n"
                "    plt.figure(figsize=(8, 3))\n"
                "    plt.bar(lags, hist, width=np.median(np.diff(lags)) if lags.size > 1 else 1.0)\n"
                "    plt.title(f\"Autocorrelogram | {channel_name_by_idx.get(int(ch), ch)}\")\n"
                "    plt.xlabel('Lag (ms)'); plt.ylabel('Count'); plt.show()\n"
            ))
        if "cross" in cfg["blocks"]:
            cells.append(self._cell(
                "if len(channel_indexes) >= 2:\n"
                "    a, b = channel_indexes[0], channel_indexes[1]\n"
                "    lags, hist = crosscorrelogram(spike_time_to_seconds(spikes_by_channel.get(int(a), [])), spike_time_to_seconds(spikes_by_channel.get(int(b), [])), sample_rate)\n"
                "    plt.figure(figsize=(8, 3)); plt.bar(lags, hist, width=np.median(np.diff(lags)) if lags.size > 1 else 1.0)\n"
                "    plt.title(f\"Crosscorrelogram | {a} vs {b}\"); plt.xlabel('Lag (ms)'); plt.ylabel('Count'); plt.show()\n"
            ))
        if "csd" in cfg["blocks"]:
            cells.append(self._cell(
                "if channel_indexes and spikes_by_channel:\n"
                "    st = spike_time_to_seconds(spikes_by_channel.get(int(channel_indexes[0]), []))\n"
                "    csd = avg_spike_csd_classic(processed, st, sample_rate, np.arange(processed.shape[0]) * 50.0)\n"
                "    plt.figure(figsize=(10, 4))\n"
                "    if csd['csd'].size:\n"
                "        plt.imshow(csd['csd'], aspect='auto', origin='lower', cmap='coolwarm')\n"
                "    plt.title('Power/source density'); plt.xlabel('Time samples'); plt.ylabel('Channel row'); plt.colorbar(); plt.show()\n"
            ))

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dump_to_file(notebook_path, notebook)
        yield {"progress": 100, "message": f"Notebook generated: {notebook_path.name}"}
