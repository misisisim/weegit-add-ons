from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.add_on_organoids.organoids_common import (
    BurstInterval,
    BurstsPayload,
    NIRBaseAddOn,
    NetworkBurstInterval,
    NetworkBurstsPayload,
)


class NetworkBurstDetectorAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, add_on_data_dir: Path) -> Optional[Tuple[Path, int, float]]:
        burst_dirs = self.list_threshold_dirs_for_module(add_on_data_dir, "burst_detector")
        selected_burst_dir = self.choose_threshold_dir_dialog(
            "Network burst detector",
            burst_dirs,
            selected_dir=self.state.selected_burst_threshold_dir,
            label="Source bursts:",
        )
        if selected_burst_dir is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Network burst detector")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        min_active_spin = QSpinBox()
        min_active_spin.setRange(2, 2048)
        min_active_spin.setValue(int(self.state.network_burst_min_active_channels))
        form.addRow("Min active channels:", min_active_spin)

        merge_gap_spin = QDoubleSpinBox()
        merge_gap_spin.setRange(0.0, 10_000.0)
        merge_gap_spin.setDecimals(3)
        merge_gap_spin.setValue(float(self.state.network_burst_merge_gap_ms))
        form.addRow("Merge gap (ms):", merge_gap_spin)

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

        min_active = int(min_active_spin.value())
        merge_gap = float(merge_gap_spin.value())
        self.state.selected_burst_threshold_dir = str(selected_burst_dir)
        self.state.network_burst_min_active_channels = min_active
        self.state.network_burst_merge_gap_ms = merge_gap
        return selected_burst_dir, min_active, merge_gap

    @staticmethod
    def _build_segments(
        bursts_by_channel: Dict[int, List[BurstInterval]],
        min_active_channels: int,
    ) -> List[Tuple[float, float]]:
        boundaries = set()
        for bursts in bursts_by_channel.values():
            for burst in bursts:
                boundaries.add(float(burst.start_ms))
                boundaries.add(float(burst.end_ms))
        points = sorted(boundaries)
        if len(points) < 2:
            return []
        segments: List[Tuple[float, float]] = []
        for left, right in zip(points[:-1], points[1:]):
            if right <= left:
                continue
            mid = 0.5 * (left + right)
            active = 0
            for bursts in bursts_by_channel.values():
                hit = False
                for burst in bursts:
                    if float(burst.start_ms) <= mid < float(burst.end_ms):
                        hit = True
                        break
                if hit:
                    active += 1
            if active >= min_active_channels:
                segments.append((left, right))
        return segments

    @staticmethod
    def _merge_segments(segments: List[Tuple[float, float]], merge_gap_ms: float) -> List[Tuple[float, float]]:
        if not segments:
            return []
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= float(merge_gap_ms):
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _active_channels_for_interval(
        bursts_by_channel: Dict[int, List[BurstInterval]],
        interval_start_ms: float,
        interval_end_ms: float,
    ) -> List[int]:
        channels: Set[int] = set()
        for channel_idx, bursts in bursts_by_channel.items():
            for burst in bursts:
                if float(burst.end_ms) < interval_start_ms or float(burst.start_ms) > interval_end_ms:
                    continue
                channels.add(int(channel_idx))
                break
        return sorted(channels)

    def run(self, session_manager, add_on_data_dir):
        params = self._ask_parameters(Path(add_on_data_dir))
        if params is None:
            return
        selected_burst_dir, min_active_channels, merge_gap_ms = params
        self.state.selected_network_burst_threshold_dir = str(Path(add_on_data_dir) / selected_burst_dir.name)

        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        bursts_path = Path(selected_burst_dir) / f"{sweep_idx}.bursts.json"
        if not bursts_path.exists():
            QMessageBox.warning(None, "Network burst detector", "No burst payload for current sweep.")
            return
        bursts_payload = BurstsPayload.model_validate_json(bursts_path.read_text(encoding="utf-8"))
        bursts_by_channel = bursts_payload.bursts_by_channel
        if not bursts_by_channel:
            QMessageBox.warning(None, "Network burst detector", "No channel bursts in selected source.")
            return

        yield {"progress": 20, "message": "Building candidate intervals..."}
        segments = self._build_segments(bursts_by_channel, min_active_channels=min_active_channels)
        merged = self._merge_segments(segments, merge_gap_ms=merge_gap_ms)

        network_bursts = []
        for i, (start_ms, end_ms) in enumerate(merged, start=1):
            active_channels = self._active_channels_for_interval(bursts_by_channel, start_ms, end_ms)
            network_bursts.append(
                NetworkBurstInterval(
                    start_ms=float(start_ms),
                    end_ms=float(end_ms),
                    active_channels=active_channels,
                )
            )
            if merged:
                yield {"progress": 20 + int((i / len(merged)) * 70), "message": f"Processed interval {i}/{len(merged)}"}

        payload = NetworkBurstsPayload(
            sweep_idx=sweep_idx,
            source_threshold_dir=str(selected_burst_dir),
            min_active_channels=min_active_channels,
            merge_gap_ms=merge_gap_ms,
            network_bursts=network_bursts,
        )
        out_dir = Path(add_on_data_dir) / selected_burst_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sweep_idx}.network_bursts.json"
        out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        yield {"progress": 100, "message": f"Saved network bursts: {out_path.name}"}
