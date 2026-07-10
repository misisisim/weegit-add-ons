from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
from pydantic import BaseModel, Field
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from weegit.core.conversions.filters import (
    BaseFilter,
    ButterworthBandPassFilter,
    ButterworthHighPassFilter,
    ButterworthLowPassFilter,
    ChebyshevBandPassFilter,
    NotchFilter,
    all_filter_names,
    filter_class_by_name,
)


class SpikePoint(BaseModel):
    time_ms: float
    value: float


class SpikesPayload(BaseModel):
    threshold: float
    sweep_idx: int
    sample_rate: float
    detector_filter: Optional[Dict[str, Any]] = None
    ignore_event_names: List[str] = Field(default_factory=list)
    ignore_before_ms: float = 0.0
    ignore_after_ms: float = 0.0
    detect_upward_spikes: bool = False
    spikes_by_channel: Dict[int, List[SpikePoint]] = Field(default_factory=dict)


@dataclass
class SharedAddOnState:
    group_idx: int = 0
    channel_indexes: List[int] = field(default_factory=list)
    ignore_event_names: List[str] = field(default_factory=list)
    ignore_before_ms: float = 0.0
    ignore_after_ms: float = 0.0
    output_folder: str = ""
    detector_threshold: float = 5.0
    detector_detect_upward_spikes: bool = False
    selected_detector_threshold_dir: str = ""
    detector_filter: Optional[Dict[str, Any]] = None
    action_potential_filter: Optional[Dict[str, Any]] = None
    local_field_filter: Optional[Dict[str, Any]] = None
    notebook_filter: Optional[Dict[str, Any]] = None
    spectrogram_filter: Optional[Dict[str, Any]] = None
    spectrogram_freq_from_hz: float = 0.0
    spectrogram_freq_to_hz: float = 300.0
    spectrogram_transform: str = "STFT"
    spectrogram_show_image: bool = True
    signal_plot_raw_enabled: bool = True
    signal_plot_ap_enabled: bool = True
    signal_plot_lfp_enabled: bool = False
    signal_plot_x_axis: bool = True
    signal_plot_y_axis: bool = True
    signal_plot_scale_bar_x: bool = False
    signal_plot_scale_bar_y: bool = False
    signal_plot_title: bool = True
    signal_plot_show_image: bool = True
    aligned_plot_x_axis: bool = True
    aligned_plot_y_axis: bool = True
    aligned_plot_background_mesh: bool = True
    aligned_plot_middle_line: bool = True
    aligned_plot_scale_bar_x: bool = False
    aligned_plot_scale_bar_y: bool = False
    aligned_plot_title: bool = True
    aligned_plot_show_image: bool = True
    raster_plot_show_image: bool = True
    plots_window_from_ms: float = 0.0
    plots_window_to_ms: float = 0.0
    burst_method: str = "max_interval"
    burst_max_isi_ms: float = 100.0
    burst_min_spikes: int = 5
    burst_min_duration_ms: float = 10.0
    network_burst_min_active_channels: int = 4
    network_burst_merge_gap_ms: float = 50.0
    reverberation_max_gap_ms: float = 30.0
    reverberation_min_packets: int = 2
    sttc_dt_ms: float = 10.0
    sttc_bin_ms: float = 5.0
    selected_burst_threshold_dir: str = ""
    selected_network_burst_threshold_dir: str = ""
    selected_connectivity_threshold_dir: str = ""
    selected_reverberation_threshold_dir: str = ""
    selected_sfc_threshold_dir: str = ""
    selected_propagation_threshold_dir: str = ""
    notebook_include_burst_detector: bool = True
    notebook_include_network_burst_detector: bool = True
    notebook_include_synchrony_connectivity: bool = True
    notebook_include_reverberation_detector: bool = True
    notebook_include_spike_field_coupling: bool = True
    notebook_include_propagation_leader_analysis: bool = True


SHARED_STATE = SharedAddOnState()


class IgnoreEventsRule(BaseModel):
    event_names: List[str]
    before_ms: float
    after_ms: float


class BurstInterval(BaseModel):
    start_ms: float
    end_ms: float
    n_spikes: int


class BurstsPayload(BaseModel):
    sweep_idx: int
    source_threshold_dir: str
    method: str
    max_isi_ms: float
    min_spikes: int
    min_duration_ms: float
    bursts_by_channel: Dict[int, List[BurstInterval]] = Field(default_factory=dict)


class NetworkBurstInterval(BaseModel):
    start_ms: float
    end_ms: float
    active_channels: List[int] = Field(default_factory=list)


class NetworkBurstsPayload(BaseModel):
    sweep_idx: int
    source_threshold_dir: str
    min_active_channels: int
    merge_gap_ms: float
    network_bursts: List[NetworkBurstInterval] = Field(default_factory=list)


class ReverberationEvent(BaseModel):
    channel_idx: int
    burst_start_ms: float
    burst_end_ms: float
    packets_count: int
    packet_centers_ms: List[float] = Field(default_factory=list)


def build_valid_mask(
    n_samples: int,
    sampling_rate: float,
    event_times_by_name: Optional[Dict[str, np.ndarray]] = None,
    ignore_event_rules: Optional[List[IgnoreEventsRule]] = None,
) -> np.ndarray:
    mask = np.ones(n_samples, dtype=bool)
    for rule in ignore_event_rules or []:
        for event_name in rule.event_names:
            for event_t in (event_times_by_name or {}).get(event_name, []):
                start_idx = max(0, int((event_t - rule.before_ms / 1000.0) * sampling_rate))
                end_idx = min(n_samples, int((event_t + rule.after_ms / 1000.0) * sampling_rate))
                if end_idx > start_idx:
                    mask[start_idx:end_idx] = False
    return mask


def detect_burst_intervals_from_spikes(
    spike_times_seconds: np.ndarray,
    *,
    method: str,
    max_isi_ms: float,
    min_spikes: int,
    min_duration_ms: float,
) -> List[Tuple[int, int]]:
    if spike_times_seconds.size < max(2, int(min_spikes)):
        return []
    times = np.asarray(spike_times_seconds, dtype=float)
    times = times[np.isfinite(times)]
    if times.size < max(2, int(min_spikes)):
        return []
    times.sort()
    isi = np.diff(times)
    max_isi_s = max(1e-6, float(max_isi_ms) / 1000.0)
    if method == "log_isi" and isi.size > 0:
        safe_isi = np.clip(isi, 1e-6, None)
        dynamic = float(np.exp(np.percentile(np.log(safe_isi), 35)))
        max_isi_s = min(max_isi_s, max(1e-6, dynamic))

    result: List[Tuple[int, int]] = []
    start_idx = 0
    for i, dt in enumerate(isi):
        if dt <= max_isi_s:
            continue
        end_idx = i
        count = end_idx - start_idx + 1
        duration_s = times[end_idx] - times[start_idx]
        if count >= int(min_spikes) and duration_s * 1000.0 >= float(min_duration_ms):
            result.append((start_idx, end_idx))
        start_idx = i + 1
    end_idx = times.size - 1
    count = end_idx - start_idx + 1
    duration_s = times[end_idx] - times[start_idx]
    if count >= int(min_spikes) and duration_s * 1000.0 >= float(min_duration_ms):
        result.append((start_idx, end_idx))
    return result


def safe_threshold_dir_name(threshold: float) -> str:
    as_text = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
    as_text = as_text.replace("-", "m").replace(".", "_")
    return f"threshold_{as_text}"


def safe_file_name(raw: str) -> str:
    clean = (raw or "").strip()
    if not clean:
        return "channel"
    for ch in '<>:"/\\|?*':
        clean = clean.replace(ch, "_")
    clean = clean.replace(" ", "_")
    return clean


class FilterEditor:
    def __init__(
        self,
        form: QFormLayout,
        title: str,
        default_spec: Optional[Dict[str, Any]],
    ):
        self._use_filter_checkbox = QCheckBox("Use filter")
        self._selector = QComboBox()
        self._selector.addItems(all_filter_names())
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._param_inputs: Dict[str, Any] = {}

        form.addRow(f"{title}:", self._use_filter_checkbox)
        form.addRow(f"{title} type:", self._selector)
        form.addRow(f"{title} params:", self._params_container)

        self._use_filter_checkbox.stateChanged.connect(self._on_filter_enabled_changed)
        self._selector.currentIndexChanged.connect(self._rebuild_filter_params)

        self.apply_spec(default_spec)

    def apply_spec(self, filter_spec: Optional[Dict[str, Any]]) -> None:
        if not filter_spec:
            self._use_filter_checkbox.setChecked(False)
            self._selector.setCurrentIndex(0)
            self._selector.setEnabled(False)
            self._rebuild_filter_params()
            return
        self._use_filter_checkbox.setChecked(bool(filter_spec.get("enabled", True)))
        filter_name = str(filter_spec.get("filter_name", "")).strip()
        idx = self._selector.findText(filter_name)
        if idx >= 0:
            self._selector.setCurrentIndex(idx)
        self._selector.setEnabled(self._use_filter_checkbox.isChecked())
        self._rebuild_filter_params()
        params = dict(filter_spec.get("params", {}) or {})
        for key, widget in self._param_inputs.items():
            if key not in params:
                continue
            value = params[key]
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))

    def _clear_layout(self, layout_obj) -> None:
        while layout_obj.count():
            item = layout_obj.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _add_param_spin(
        self, label: str, key: str, value: float, min_v: float, max_v: float, step: float
    ) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setDecimals(6)
        spin.setSingleStep(step)
        spin.setValue(float(value))
        row.addWidget(spin)
        self._params_layout.addLayout(row)
        self._param_inputs[key] = spin

    def _add_param_int(self, label: str, key: str, value: int, min_v: int, max_v: int) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(1)
        spin.setValue(int(value))
        row.addWidget(spin)
        self._params_layout.addLayout(row)
        self._param_inputs[key] = spin

    def _on_filter_enabled_changed(self, _state: int) -> None:
        self._selector.setEnabled(self._use_filter_checkbox.isChecked())
        self._rebuild_filter_params()

    def _rebuild_filter_params(self) -> None:
        self._clear_layout(self._params_layout)
        self._param_inputs.clear()
        if not self._use_filter_checkbox.isChecked():
            return
        cls = filter_class_by_name(self._selector.currentText().strip())
        if cls is None:
            return
        flt = cls()
        if isinstance(flt, ButterworthLowPassFilter):
            self._add_param_spin("Cutoff (Hz):", "cutoff_hz", flt.cutoff_hz, 0.1, 1e6, 1.0)
            self._add_param_int("Order:", "order", flt.order, 1, 12)
        elif isinstance(flt, ButterworthHighPassFilter):
            self._add_param_spin("Cutoff (Hz):", "cutoff_hz", flt.cutoff_hz, 0.1, 1e6, 1.0)
            self._add_param_int("Order:", "order", flt.order, 1, 12)
        elif isinstance(flt, ButterworthBandPassFilter):
            self._add_param_spin("Low cut (Hz):", "lowcut_hz", flt.lowcut_hz, 0.1, 1e6, 1.0)
            self._add_param_spin("High cut (Hz):", "highcut_hz", flt.highcut_hz, 0.1, 1e6, 1.0)
            self._add_param_int("Order:", "order", flt.order, 1, 12)
        elif isinstance(flt, ChebyshevBandPassFilter):
            self._add_param_spin("Low cut (Hz):", "lowcut_hz", flt.lowcut_hz, 0.1, 1e6, 1.0)
            self._add_param_spin("High cut (Hz):", "highcut_hz", flt.highcut_hz, 0.1, 1e6, 1.0)
            self._add_param_int("Order:", "order", flt.order, 1, 12)
            self._add_param_spin("Ripple (dB):", "ripple_db", flt.ripple_db, 0.1, 10.0, 0.1)
        elif isinstance(flt, NotchFilter):
            self._add_param_spin("Notch (Hz):", "notch_freq_hz", flt.notch_freq_hz, 1.0, 1e6, 1.0)
            self._add_param_spin("Q factor:", "q_factor", flt.q_factor, 0.1, 200.0, 0.1)

    def get_filter_spec(self) -> Optional[Dict[str, Any]]:
        if not self._use_filter_checkbox.isChecked():
            return None
        params: Dict[str, Any] = {}
        for key, widget in self._param_inputs.items():
            if isinstance(widget, QDoubleSpinBox):
                params[key] = float(widget.value())
            elif isinstance(widget, QSpinBox):
                params[key] = int(widget.value())
        return {
            "enabled": True,
            "filter_name": self._selector.currentText().strip(),
            "params": params,
        }


def filter_from_spec(spec: Optional[Dict[str, Any]]) -> Optional[BaseFilter]:
    if not spec:
        return None
    if not bool(spec.get("enabled", True)):
        return None
    cls = filter_class_by_name(str(spec.get("filter_name", "")).strip())
    if cls is None:
        return None
    flt = cls()
    setattr(flt, "enabled", True)
    for key, value in (spec.get("params", {}) or {}).items():
        if hasattr(flt, key):
            current = getattr(flt, key)
            if isinstance(current, int):
                setattr(flt, key, int(value))
            else:
                setattr(flt, key, float(value))
    if hasattr(flt, "sos_cache"):
        flt.sos_cache = {}
    return flt


class NIRBaseAddOn:

    @property
    def state(self) -> SharedAddOnState:
        return SHARED_STATE

    def non_aux_groups(self, session_manager) -> List[Tuple[int, Any]]:
        return [
            (idx, group)
            for idx, group in enumerate(session_manager.user_session.gui_setup.channels_groups or [])
            if (not getattr(group, "is_auxiliary", False)) and getattr(group, "channel_indexes", [])
        ]

    def ensure_non_aux_groups(self, session_manager, title: str) -> Optional[List[Tuple[int, Any]]]:
        groups = self.non_aux_groups(session_manager)
        if not groups:
            QMessageBox.warning(None, title, "No non-auxiliary channel groups configured.")
            return None
        return groups

    def event_vocabulary_names(self, session_manager) -> List[str]:
        names: List[str] = []
        for _event_id, entry in (session_manager.events_vocabulary or {}).items():
            name = str(getattr(entry, "name", "")).strip()
            if name:
                names.append(name)
        return sorted(set(names))

    def event_times_by_name_for_window(
        self,
        session_manager,
        sweep_idx: int,
        start_second: float,
        end_second: float,
    ) -> Dict[str, np.ndarray]:
        event_times: Dict[str, List[float]] = {}
        for event in session_manager.events or []:
            if int(event.sweep_idx) != int(sweep_idx):
                continue
            event_time_s = float(event.time_ms) / 1000.0
            if event_time_s < start_second or event_time_s > end_second:
                continue
            event_name = session_manager.user_session.get_event_vocabulary_name(event.event_name_id)
            event_times.setdefault(event_name, []).append(event_time_s - start_second)
        return {k: np.asarray(v, dtype=float) for k, v in event_times.items()}

    def build_group_channel_selector(
        self,
        form: QFormLayout,
        groups: List[Tuple[int, Any]],
        header,
    ) -> Tuple[QComboBox, QListWidget]:
        group_combo = QComboBox()
        for group_idx, group in groups:
            group_combo.addItem(f"#{group_idx} {group.name}", group_idx)
        default_group_pos = group_combo.findData(int(self.state.group_idx))
        group_combo.setCurrentIndex(max(0, default_group_pos))
        form.addRow("Channel group:", group_combo)

        channels_list = QListWidget()
        channels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        form.addRow("Channels:", channels_list)

        def rebuild() -> None:
            channels_list.clear()
            group_idx = int(group_combo.currentData())
            group = next((g for idx, g in groups if idx == group_idx), None)
            enabled = set(getattr(group, "enabled_indexes", set()) or set()) if group is not None else set()
            preferred = set(self.state.channel_indexes or [])
            selected_by_default = preferred if preferred else enabled
            for ch_idx in (getattr(group, "channel_indexes", []) or []):
                ch_name = ""
                try:
                    if 0 <= ch_idx < len(header.channel_info.name):
                        ch_name = header.channel_info.name[ch_idx]
                except Exception:
                    ch_name = ""
                label = f"{ch_idx}" if not ch_name else f"{ch_idx} [{ch_name}]"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, int(ch_idx))
                item.setSelected(int(ch_idx) in selected_by_default)
                channels_list.addItem(item)

        group_combo.currentIndexChanged.connect(lambda _idx: rebuild())
        rebuild()
        return group_combo, channels_list

    def selected_channels(self, channels_list: QListWidget) -> List[int]:
        selected: List[int] = []
        seen = set()
        for row in range(channels_list.count()):
            item = channels_list.item(row)
            if item is None or (not item.isSelected()):
                continue
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is None:
                continue
            ch_idx = int(value)
            if ch_idx in seen:
                continue
            selected.append(ch_idx)
            seen.add(ch_idx)
        return selected

    @staticmethod
    def channel_name(header, channel_idx: int) -> str:
        try:
            names = header.channel_info.name or []
            if 0 <= int(channel_idx) < len(names):
                name = str(names[int(channel_idx)]).strip()
                if name:
                    return name
        except Exception:
            pass
        return f"ch{int(channel_idx)}"

    def build_ignore_events_controls(
        self,
        form: QFormLayout,
        session_manager,
    ) -> Tuple[QListWidget, QDoubleSpinBox, QDoubleSpinBox]:
        events_list = QListWidget()
        events_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        selected_names = set(self.state.ignore_event_names or [])
        for name in self.event_vocabulary_names(session_manager):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setSelected(name in selected_names)
            events_list.addItem(item)
        form.addRow("Ignore events:", events_list)

        before_spin = QDoubleSpinBox()
        before_spin.setRange(0.0, 60_000.0)
        before_spin.setDecimals(3)
        before_spin.setValue(float(self.state.ignore_before_ms))
        form.addRow("Ignore before (ms):", before_spin)

        after_spin = QDoubleSpinBox()
        after_spin.setRange(0.0, 60_000.0)
        after_spin.setDecimals(3)
        after_spin.setValue(float(self.state.ignore_after_ms))
        form.addRow("Ignore after (ms):", after_spin)
        return events_list, before_spin, after_spin

    def selected_ignore_event_names(self, events_list: QListWidget) -> List[str]:
        names: List[str] = []
        for item in events_list.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if value:
                names.append(str(value))
        return sorted(set(names))

    def persist_common_selection(
        self,
        group_idx: int,
        channels: List[int],
        ignore_event_names: List[str],
        ignore_before_ms: float,
        ignore_after_ms: float,
    ) -> None:
        self.state.group_idx = int(group_idx)
        self.state.channel_indexes = list(channels)
        self.state.ignore_event_names = list(ignore_event_names)
        self.state.ignore_before_ms = float(ignore_before_ms)
        self.state.ignore_after_ms = float(ignore_after_ms)

    def detector_dir_from_any_add_on_dir(self, add_on_data_dir: Path) -> Path:
        prefix = "dev_" if Path(add_on_data_dir).name.startswith("dev_") else ""
        return Path(add_on_data_dir).parent / f"{prefix}spikes_detector"

    def module_dir_from_any_add_on_dir(self, add_on_data_dir: Path, module_name: str) -> Path:
        prefix = "dev_" if Path(add_on_data_dir).name.startswith("dev_") else ""
        return Path(add_on_data_dir).parent / f"{prefix}{module_name}"

    def list_detector_threshold_dirs(self, add_on_data_dir: Path) -> List[Path]:
        detector_dir = self.detector_dir_from_any_add_on_dir(add_on_data_dir)
        if not detector_dir.exists():
            return []
        result: List[Path] = []
        for child in detector_dir.iterdir():
            if child.is_dir() and child.name.startswith("threshold_"):
                result.append(child)
        return sorted(result, key=lambda p: p.name)

    def list_threshold_dirs_for_module(self, add_on_data_dir: Path, module_name: str) -> List[Path]:
        base_dir = self.module_dir_from_any_add_on_dir(add_on_data_dir, module_name)
        if not base_dir.exists():
            return []
        result: List[Path] = []
        for child in base_dir.iterdir():
            if child.is_dir() and child.name.startswith("threshold_"):
                result.append(child)
        return sorted(result, key=lambda p: p.name)

    def choose_threshold_dir_dialog(
        self,
        title: str,
        threshold_dirs: List[Path],
        selected_dir: str = "",
        label: str = "Detected spikes:",
    ) -> Optional[Path]:
        if not threshold_dirs:
            QMessageBox.warning(None, title, "No spikes detected yet. Run spikes detector first.")
            return None
        dialog = QDialog()
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        combo = QComboBox()
        for path in threshold_dirs:
            combo.addItem(path.name, str(path))
        default_idx = 0
        current_selected = selected_dir.strip() if selected_dir else self.state.selected_detector_threshold_dir
        if current_selected:
            matched = combo.findData(current_selected)
            if matched >= 0:
                default_idx = matched
        combo.setCurrentIndex(default_idx)
        form.addRow(label, combo)
        layout.addLayout(form)

        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Select")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        selected = Path(str(combo.currentData()))
        self.state.selected_detector_threshold_dir = str(selected)
        return selected

    def read_spikes_payload(self, threshold_dir: Path, sweep_idx: int) -> Optional[SpikesPayload]:
        spikes_path = Path(threshold_dir) / f"{int(sweep_idx)}.spikes.json"
        if not spikes_path.exists():
            return None
        return SpikesPayload.model_validate_json(spikes_path.read_text(encoding="utf-8"))

    def read_json_payload(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def save_payload(path: Path, payload: SpikesPayload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def save_figure(fig, save_path: Optional[Path], show: bool) -> None:
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), dpi=200, bbox_inches="tight")
        if show:
            fig.show()

    @staticmethod
    def json_dump_to_file(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
