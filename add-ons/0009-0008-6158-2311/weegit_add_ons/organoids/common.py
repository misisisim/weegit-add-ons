from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import numpy as np
from pydantic import BaseModel, Field

from weegit_add_ons.add_on_organoids.organoids_common import (
    FilterEditor,
    IgnoreEventsRule,
    NIRBaseAddOn as LegacyNIRBaseAddOn,
    build_valid_mask,
    filter_from_spec,
    safe_file_name,
)


PIPELINES_FILENAME = "preprocessing_pipelines.json"
DEFAULT_PIPELINE_NAME = "raw"


class SpikePoint(BaseModel):
    sample_idx: Optional[int] = None
    time_ms: float
    value: float
    value_uv: Optional[float] = None
    polarity: str = "negative"


class SpikesPayload(BaseModel):
    detector_name: str = "adaptive_mad"
    preprocessing_pipeline: str = DEFAULT_PIPELINE_NAME
    threshold: float = 6.0
    sweep_idx: int
    sample_rate: float
    adaptive_sigma: bool = True
    sigma_params: Dict[str, Any] = Field(default_factory=dict)
    detect_positive: bool = False
    detect_negative: bool = True
    merge_window_ms: float = 1.0
    ignore_event_names: List[str] = Field(default_factory=list)
    ignore_before_ms: float = 0.0
    ignore_after_ms: float = 0.0
    spikes_by_channel: Dict[int, List[SpikePoint]] = Field(default_factory=dict)


@dataclass
class OrganoidsSharedState:
    group_idx: int = 0
    channel_indexes: List[int] = field(default_factory=list)
    ignore_event_names: List[str] = field(default_factory=list)
    ignore_before_ms: float = 0.0
    ignore_after_ms: float = 0.0
    output_folder: str = ""
    selected_pipeline_name: str = DEFAULT_PIPELINE_NAME
    selected_spikes_dir: str = ""
    detector_threshold: float = 6.0
    detector_adaptive_sigma: bool = True
    detector_detect_positive: bool = False
    detector_detect_negative: bool = True
    detector_merge_window_ms: float = 1.0
    sigma_window_ms: float = 500.0
    sigma_step_ms: float = 100.0
    sigma_floor_uv: float = 2.0
    sigma_smooth_windows: int = 3
    plots_window_from_ms: float = 0.0
    plots_window_to_ms: float = 0.0
    visualization_show_image: bool = True
    visualization_freq_from_hz: float = 0.0
    visualization_freq_to_hz: float = 300.0
    visualization_include_signal: bool = True
    visualization_include_spectrogram: bool = False
    visualization_include_csd: bool = False
    visualization_include_aligned_spikes: bool = False
    visualization_include_raster: bool = True
    visualization_include_autocorrelogram: bool = False
    visualization_include_crosscorrelogram: bool = False
    notebook_include_signal: bool = True
    notebook_include_spectrogram: bool = True
    notebook_include_csd: bool = False
    notebook_include_aligned_spikes: bool = True
    notebook_include_raster: bool = True
    notebook_include_autocorrelogram: bool = True
    notebook_include_crosscorrelogram: bool = False


SHARED_STATE = OrganoidsSharedState()


class OrganoidsBaseAddOn(LegacyNIRBaseAddOn):
    @property
    def state(self) -> OrganoidsSharedState:
        return SHARED_STATE

    def pipelines_path(self, add_on_data_dir: Path) -> Path:
        return Path(add_on_data_dir).parent / "preprocessing_pipeline" / PIPELINES_FILENAME

    def spikes_detection_dir(self, add_on_data_dir: Path) -> Path:
        prefix = "dev_" if Path(add_on_data_dir).name.startswith("dev_") else ""
        return Path(add_on_data_dir).parent / f"{prefix}spike_detection"

    def list_spike_detection_dirs(self, add_on_data_dir: Path) -> List[Path]:
        base = self.spikes_detection_dir(add_on_data_dir)
        if not base.exists():
            return []
        return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)

    def read_spikes_payload(self, spikes_dir: Path, sweep_idx: int) -> Optional[SpikesPayload]:
        path = Path(spikes_dir) / f"{int(sweep_idx)}.spikes.json"
        if not path.exists():
            return None
        return SpikesPayload.model_validate_json(path.read_text(encoding="utf-8"))

    def choose_spikes_dir_dialog(
        self,
        title: str,
        add_on_data_dir: Path,
        label: str = "Detected spikes:",
    ) -> Optional[Path]:
        from PyQt6.QtWidgets import QComboBox, QDialog, QFormLayout, QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout

        dirs = self.list_spike_detection_dirs(add_on_data_dir)
        if not dirs:
            QMessageBox.warning(None, title, "No detected spikes yet. Run spike_detection first.")
            return None
        dialog = QDialog()
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        combo = QComboBox()
        for path in dirs:
            combo.addItem(path.name, str(path))
        if self.state.selected_spikes_dir:
            idx = combo.findData(self.state.selected_spikes_dir)
            combo.setCurrentIndex(max(0, idx))
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
        self.state.selected_spikes_dir = str(selected)
        return selected

    @staticmethod
    def save_spikes_payload(path: Path, payload: SpikesPayload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def json_dump_to_file(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    @staticmethod
    def channel_matrix_from_session(
        session_manager,
        channel_indexes: List[int],
        sweep_idx: int,
        start_sample: int,
        end_sample: int,
        sample_rate: float,
    ) -> np.ndarray:
        n_samples = max(1, int(end_sample) - int(start_sample))
        rows = []
        for channel_idx in channel_indexes:
            signal = session_manager.experiment_data.process_single_channel(
                channel_idx=int(channel_idx),
                sweep_idx=int(sweep_idx),
                start_sample=int(start_sample),
                end_sample=int(end_sample),
                each_point=1,
                sample_rate=float(sample_rate),
                filters=[],
                output_number_of_dots=n_samples,
                transformation_add_ons=[],
            )
            rows.append(np.asarray(signal, dtype=np.float64))
        return np.vstack(rows) if rows else np.empty((0, n_samples), dtype=np.float64)


__all__ = [
    "DEFAULT_PIPELINE_NAME",
    "PIPELINES_FILENAME",
    "FilterEditor",
    "IgnoreEventsRule",
    "OrganoidsBaseAddOn",
    "SpikePoint",
    "SpikesPayload",
    "build_valid_mask",
    "filter_from_spec",
    "safe_file_name",
]
