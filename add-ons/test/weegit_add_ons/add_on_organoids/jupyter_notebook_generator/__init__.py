from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from weegit_add_ons.add_on_organoids.organoids_common import FilterEditor, NIRBaseAddOn


class JupyterNotebookGeneratorAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header, add_on_data_dir: Path
    ) -> Optional[Tuple[str, List[int], int, Optional[dict], Path, Dict[str, bool]]]:
        groups = self.ensure_non_aux_groups(session_manager, "Jupyter notebook generator")
        if groups is None:
            return None
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_threshold_dir = self.choose_threshold_dir_dialog("Jupyter notebook generator", threshold_dirs)
        if selected_threshold_dir is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Jupyter notebook generator")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)
        filter_editor = FilterEditor(form, "Filter", self.state.notebook_filter)

        cb_burst = QCheckBox("Include Burst Detector results")
        cb_burst.setChecked(bool(self.state.notebook_include_burst_detector))
        form.addRow("Add-ons:", cb_burst)
        cb_network = QCheckBox("Include Network Burst Detector results")
        cb_network.setChecked(bool(self.state.notebook_include_network_burst_detector))
        form.addRow("", cb_network)
        cb_connectivity = QCheckBox("Include Synchrony & Connectivity results")
        cb_connectivity.setChecked(bool(self.state.notebook_include_synchrony_connectivity))
        form.addRow("", cb_connectivity)
        cb_reverb = QCheckBox("Include Reverberation Detector results")
        cb_reverb.setChecked(bool(self.state.notebook_include_reverberation_detector))
        form.addRow("", cb_reverb)
        cb_sfc = QCheckBox("Include Spike-Field Coupling results")
        cb_sfc.setChecked(bool(self.state.notebook_include_spike_field_coupling))
        form.addRow("", cb_sfc)
        cb_prop = QCheckBox("Include Propagation & Leader Analysis results")
        cb_prop.setChecked(bool(self.state.notebook_include_propagation_leader_analysis))
        form.addRow("", cb_prop)

        path_edit = QLineEdit()
        default_name = f"analysis_sweep_{int(session_manager.gui_setup.current_sweep_idx)}.ipynb"
        if self.state.output_folder:
            path_edit.setText(str(Path(self.state.output_folder) / default_name))
        else:
            path_edit.setText(default_name)
        browse_btn = QPushButton("Browse")
        row = QHBoxLayout()
        row.addWidget(path_edit, 1)
        row.addWidget(browse_btn)
        form.addRow("Notebook path:", row)

        def choose_notebook_path() -> None:
            current = path_edit.text().strip() or default_name
            selected, _ = QFileDialog.getSaveFileName(
                dialog,
                "Select notebook path",
                current,
                "Jupyter Notebook (*.ipynb)",
            )
            if selected:
                if not selected.endswith(".ipynb"):
                    selected += ".ipynb"
                path_edit.setText(selected)

        browse_btn.clicked.connect(choose_notebook_path)

        layout.addLayout(form)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_generate = QPushButton("Generate")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_generate)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_generate.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        channels = self.selected_channels(channels_list)
        if not channels:
            QMessageBox.warning(dialog, "Jupyter notebook generator", "Select at least one channel.")
            return None
        notebook_path = path_edit.text().strip()
        if not notebook_path:
            QMessageBox.warning(dialog, "Jupyter notebook generator", "Select notebook output path.")
            return None

        self.state.notebook_filter = filter_editor.get_filter_spec()
        self.state.notebook_include_burst_detector = cb_burst.isChecked()
        self.state.notebook_include_network_burst_detector = cb_network.isChecked()
        self.state.notebook_include_synchrony_connectivity = cb_connectivity.isChecked()
        self.state.notebook_include_reverberation_detector = cb_reverb.isChecked()
        self.state.notebook_include_spike_field_coupling = cb_sfc.isChecked()
        self.state.notebook_include_propagation_leader_analysis = cb_prop.isChecked()
        self.state.output_folder = str(Path(notebook_path).parent)
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=self.state.ignore_event_names,
            ignore_before_ms=self.state.ignore_before_ms,
            ignore_after_ms=self.state.ignore_after_ms,
        )
        include_flags = {
            "burst": cb_burst.isChecked(),
            "network_burst": cb_network.isChecked(),
            "connectivity": cb_connectivity.isChecked(),
            "reverberation": cb_reverb.isChecked(),
            "sfc": cb_sfc.isChecked(),
            "propagation": cb_prop.isChecked(),
        }
        return notebook_path, channels, int(group_combo.currentData()), filter_editor.get_filter_spec(), selected_threshold_dir, include_flags

    @staticmethod
    def _cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    @staticmethod
    def _markdown(source: str) -> dict:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        notebook_path_raw, channels, _group_idx, filter_spec, threshold_dir, include_flags = params
        notebook_path = Path(notebook_path_raw)

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        payload = self.read_spikes_payload(threshold_dir, sweep_idx)
        if payload is None:
            QMessageBox.warning(None, "Jupyter notebook generator", "No spikes file for current sweep in selected threshold folder.")
            return

        experiment_folder = Path(session_manager.weegit_experiment_folder)
        session_filename = getattr(session_manager.user_session, "session_filename", "")
        filter_block = repr(filter_spec or {})
        channel_name_map = {int(ch): self.channel_name(header, int(ch)) for ch in channels}
        threshold_name = threshold_dir.name
        add_on_data_root = Path(add_on_data_dir).parent
        burst_path = add_on_data_root / "dev_burst_detector" / threshold_name / f"{sweep_idx}.bursts.json"
        network_burst_path = add_on_data_root / "dev_network_burst_detector" / threshold_name / f"{sweep_idx}.network_bursts.json"
        connectivity_path = add_on_data_root / "dev_synchrony_connectivity" / threshold_name / f"{sweep_idx}.connectivity.json"
        reverberation_path = add_on_data_root / "dev_reverberation_detector" / threshold_name / f"{sweep_idx}.reverberations.json"
        sfc_path = add_on_data_root / "dev_spike_field_coupling" / threshold_name / f"{sweep_idx}.sfc.json"
        propagation_path = add_on_data_root / "dev_propagation_leader_analysis" / threshold_name / f"{sweep_idx}.propagation_leaders.json"

        cells = [
            self._markdown(
                "# Weegit analysis notebook\n"
                "Auto-generated notebook for quick interactive analysis.\n"
            ),
            self._cell(
                "from pathlib import Path\n"
                "import json\n"
                "import numpy as np\n"
                "import matplotlib.pyplot as plt\n"
                "from weegit.core.weegit_session import WeegitSessionManager\n"
                "from weegit.core.conversions.filters import filter_class_by_name\n"
            ),
            self._cell(
                f"weegit_folder = Path(r'''{str(experiment_folder)}''')\n"
                f"session_filename = r'''{session_filename}'''\n"
                f"sweep_idx = {sweep_idx}\n"
                f"channel_indexes = {channels}\n"
                f"channel_name_by_idx = {channel_name_map}\n"
                f"detected_threshold = {float(payload.threshold)}\n"
                f"spikes_path = Path(r'''{str(threshold_dir / f'{sweep_idx}.spikes.json')}''')\n"
                f"burst_path = Path(r'''{str(burst_path)}''')\n"
                f"network_burst_path = Path(r'''{str(network_burst_path)}''')\n"
                f"connectivity_path = Path(r'''{str(connectivity_path)}''')\n"
                f"reverberation_path = Path(r'''{str(reverberation_path)}''')\n"
                f"sfc_path = Path(r'''{str(sfc_path)}''')\n"
                f"propagation_path = Path(r'''{str(propagation_path)}''')\n"
                f"filter_spec = {filter_block}\n"
            ),
            self._cell(
                "manager = WeegitSessionManager()\n"
                "manager.init_from_folder(weegit_folder)\n"
                "if session_filename:\n"
                "    manager.switch_sessions(session_filename)\n"
                "header = manager.experiment_data.header\n"
                "sample_rate = float(header.sample_rate)\n"
            ),
            self._cell(
                "spikes_payload = json.loads(spikes_path.read_text(encoding='utf-8'))\n"
                "spikes_by_channel = {\n"
                "    int(ch): rows for ch, rows in spikes_payload.get('spikes_by_channel', {}).items()\n"
                "}\n"
                "print(f'Loaded spikes for channels: {sorted(spikes_by_channel.keys())}')\n"
            ),
            self._cell(
                "def build_filter(spec):\n"
                "    if not spec:\n"
                "        return None\n"
                "    cls = filter_class_by_name(str(spec.get('filter_name', '')).strip())\n"
                "    if cls is None:\n"
                "        return None\n"
                "    flt = cls()\n"
                "    flt.enabled = bool(spec.get('enabled', True))\n"
                "    for key, value in (spec.get('params', {}) or {}).items():\n"
                "        if hasattr(flt, key):\n"
                "            setattr(flt, key, value)\n"
                "    if hasattr(flt, 'sos_cache'):\n"
                "        flt.sos_cache = {}\n"
                "    return flt\n"
            ),
            self._cell(
                "flt = build_filter(filter_spec)\n"
                "for ch in channel_indexes:\n"
                "    sweep_points = int(header.number_of_points_per_sweep[sweep_idx])\n"
                "    signal = manager.experiment_data.process_single_channel(\n"
                "        channel_idx=int(ch),\n"
                "        sweep_idx=sweep_idx,\n"
                "        start_sample=0,\n"
                "        end_sample=sweep_points,\n"
                "        each_point=1,\n"
                "        sample_rate=sample_rate,\n"
                "        filters=[flt] if flt is not None else [],\n"
                "        output_number_of_dots=sweep_points,\n"
                "        transformation_add_ons=[],\n"
                "    )\n"
                "    signal = np.asarray(signal, dtype=np.float64)\n"
                "    times = np.arange(signal.size) / sample_rate\n"
                "    plt.figure(figsize=(14, 3))\n"
                "    plt.plot(times, signal, linewidth=0.6, color='black')\n"
                "    ch_spikes = spikes_by_channel.get(int(ch), [])\n"
                "    if ch_spikes:\n"
                "        st = np.array([row['time_ms'] for row in ch_spikes], dtype=float) / 1000.0\n"
                "        sy = np.interp(st, times, signal)\n"
                "        plt.scatter(st, sy, s=10, c='red')\n"
                "    channel_name = channel_name_by_idx.get(int(ch), f'ch{ch}')\n"
                "    plt.title(f'Channel {channel_name}')\n"
                "    plt.xlabel('Time (s)')\n"
                "    plt.ylabel('uV')\n"
                "    plt.grid(True, alpha=0.2)\n"
                "    plt.show()\n"
            ),
        ]

        if include_flags.get("burst", False):
            cells.append(self._cell(
                "if burst_path.exists():\n"
                "    burst_payload = json.loads(burst_path.read_text(encoding='utf-8'))\n"
                "    counts = {int(ch): len(rows) for ch, rows in burst_payload.get('bursts_by_channel', {}).items()}\n"
                "    print('Burst counts by channel:', counts)\n"
                "else:\n"
                "    print('Burst payload not found:', burst_path)\n"
            ))
        if include_flags.get("network_burst", False):
            cells.append(self._cell(
                "if network_burst_path.exists():\n"
                "    nb_payload = json.loads(network_burst_path.read_text(encoding='utf-8'))\n"
                "    nbs = nb_payload.get('network_bursts', [])\n"
                "    print('Network bursts:', len(nbs))\n"
                "else:\n"
                "    print('Network burst payload not found:', network_burst_path)\n"
            ))
        if include_flags.get("connectivity", False):
            cells.append(self._cell(
                "if connectivity_path.exists():\n"
                "    conn_payload = json.loads(connectivity_path.read_text(encoding='utf-8'))\n"
                "    sttc = np.asarray(conn_payload.get('sttc_matrix', []), dtype=float)\n"
                "    if sttc.size:\n"
                "        plt.figure(figsize=(6, 5))\n"
                "        plt.imshow(sttc, vmin=-1, vmax=1, cmap='coolwarm')\n"
                "        plt.title('STTC matrix')\n"
                "        plt.colorbar()\n"
                "        plt.show()\n"
                "else:\n"
                "    print('Connectivity payload not found:', connectivity_path)\n"
            ))
        if include_flags.get("reverberation", False):
            cells.append(self._cell(
                "if reverberation_path.exists():\n"
                "    rev_payload = json.loads(reverberation_path.read_text(encoding='utf-8'))\n"
                "    print('Reverberation events:', len(rev_payload.get('events', [])))\n"
                "else:\n"
                "    print('Reverberation payload not found:', reverberation_path)\n"
            ))
        if include_flags.get("sfc", False):
            cells.append(self._cell(
                "if sfc_path.exists():\n"
                "    sfc_payload = json.loads(sfc_path.read_text(encoding='utf-8'))\n"
                "    channels_data = sfc_payload.get('channels', {})\n"
                "    labels = [channels_data[k].get('channel_name', k) for k in channels_data.keys()]\n"
                "    values = [channels_data[k].get('plv', 0.0) for k in channels_data.keys()]\n"
                "    plt.figure(figsize=(10, 4))\n"
                "    plt.bar(range(len(values)), values)\n"
                "    plt.xticks(range(len(values)), labels, rotation=90)\n"
                "    plt.ylim(0, 1)\n"
                "    plt.title('Spike-field coupling (PLV)')\n"
                "    plt.show()\n"
                "else:\n"
                "    print('SFC payload not found:', sfc_path)\n"
            ))
        if include_flags.get("propagation", False):
            cells.append(self._cell(
                "if propagation_path.exists():\n"
                "    prop_payload = json.loads(propagation_path.read_text(encoding='utf-8'))\n"
                "    counts = prop_payload.get('leader_counts', {})\n"
                "    print('Leader counts:', counts)\n"
                "else:\n"
                "    print('Propagation payload not found:', propagation_path)\n"
            ))
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_dump_to_file(notebook_path, notebook)
        yield {"progress": 100, "message": f"Notebook generated: {notebook_path.name}"}
