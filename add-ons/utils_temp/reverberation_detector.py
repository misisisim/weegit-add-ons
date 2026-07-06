from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import json
import numpy as np
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from nir_common import BurstsPayload, NIRBaseAddOn, ReverberationEvent


class ReverberationDetectorAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, add_on_data_dir: Path) -> Optional[Tuple[Path, float, int]]:
        burst_dirs = self.list_threshold_dirs_for_module(add_on_data_dir, "burst_detector")
        selected_burst_dir = self.choose_threshold_dir_dialog(
            "Reverberation detector",
            burst_dirs,
            selected_dir=self.state.selected_reverberation_threshold_dir,
            label="Source bursts:",
        )
        if selected_burst_dir is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Reverberation detector")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        max_gap_spin = QDoubleSpinBox()
        max_gap_spin.setRange(1.0, 5000.0)
        max_gap_spin.setDecimals(3)
        max_gap_spin.setValue(float(self.state.reverberation_max_gap_ms))
        form.addRow("Packet max gap (ms):", max_gap_spin)

        min_packets_spin = QSpinBox()
        min_packets_spin.setRange(2, 100)
        min_packets_spin.setValue(int(self.state.reverberation_min_packets))
        form.addRow("Min packets in burst:", min_packets_spin)

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

        max_gap_ms = float(max_gap_spin.value())
        min_packets = int(min_packets_spin.value())
        self.state.reverberation_max_gap_ms = max_gap_ms
        self.state.reverberation_min_packets = min_packets
        self.state.selected_reverberation_threshold_dir = str(selected_burst_dir)
        return selected_burst_dir, max_gap_ms, min_packets

    @staticmethod
    def _split_packets(spike_times_ms: np.ndarray, max_gap_ms: float) -> List[np.ndarray]:
        if spike_times_ms.size == 0:
            return []
        if spike_times_ms.size == 1:
            return [spike_times_ms]
        packets = []
        start = 0
        diffs = np.diff(spike_times_ms)
        for i, dt in enumerate(diffs):
            if float(dt) <= float(max_gap_ms):
                continue
            packets.append(spike_times_ms[start : i + 1])
            start = i + 1
        packets.append(spike_times_ms[start:])
        return packets

    def run(self, session_manager, add_on_data_dir):
        params = self._ask_parameters(Path(add_on_data_dir))
        if params is None:
            return
        selected_burst_dir, max_gap_ms, min_packets = params
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)

        bursts_path = Path(selected_burst_dir) / f"{sweep_idx}.bursts.json"
        if not bursts_path.exists():
            QMessageBox.warning(None, "Reverberation detector", "No burst payload for current sweep.")
            return
        bursts_payload = BurstsPayload.model_validate_json(bursts_path.read_text(encoding="utf-8"))

        source_threshold_dir = Path(str(bursts_payload.source_threshold_dir))
        if not source_threshold_dir.exists():
            source_threshold_dir = self.detector_dir_from_any_add_on_dir(Path(add_on_data_dir)) / selected_burst_dir.name
        spikes_payload = self.read_spikes_payload(source_threshold_dir, sweep_idx)
        if spikes_payload is None:
            QMessageBox.warning(None, "Reverberation detector", "No source spikes payload for selected bursts.")
            return

        events: List[ReverberationEvent] = []
        channels = sorted(bursts_payload.bursts_by_channel.keys())
        total = max(1, len(channels))
        yield {"progress": 0, "message": "Detecting reverberations..."}
        for i, channel_idx in enumerate(channels, start=1):
            ch_spikes_ms = np.asarray(
                [float(point.time_ms) for point in spikes_payload.spikes_by_channel.get(int(channel_idx), [])],
                dtype=float,
            )
            ch_spikes_ms.sort()
            for burst in bursts_payload.bursts_by_channel.get(int(channel_idx), []):
                mask = (ch_spikes_ms >= float(burst.start_ms)) & (ch_spikes_ms <= float(burst.end_ms))
                in_burst = ch_spikes_ms[mask]
                packets = self._split_packets(in_burst, max_gap_ms=max_gap_ms)
                if len(packets) < int(min_packets):
                    continue
                centers = [float(np.mean(packet)) for packet in packets if packet.size]
                events.append(
                    ReverberationEvent(
                        channel_idx=int(channel_idx),
                        burst_start_ms=float(burst.start_ms),
                        burst_end_ms=float(burst.end_ms),
                        packets_count=int(len(packets)),
                        packet_centers_ms=centers,
                    )
                )
            yield {"progress": int((i / total) * 100), "message": f"Processed channel {i}/{total}"}

        out_dir = Path(add_on_data_dir) / selected_burst_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sweep_idx}.reverberations.json"
        payload = {
            "sweep_idx": sweep_idx,
            "source_burst_dir": str(selected_burst_dir),
            "source_spikes_dir": str(source_threshold_dir),
            "max_gap_ms": max_gap_ms,
            "min_packets": min_packets,
            "events": [event.model_dump() for event in events],
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        yield {"progress": 100, "message": f"Saved reverberations: {out_path.name}"}
