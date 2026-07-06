from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import json
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from weegit.core.add_ons.base import BaseAddOn

from nir_common import NIRBaseAddOn, safe_file_name


class SynchronyConnectivityAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(self, session_manager, header, add_on_data_dir: Path) -> Optional[Tuple[Path, List[int], bool, bool, str, float, float]]:
        threshold_dirs = self.list_detector_threshold_dirs(add_on_data_dir)
        selected_source_dir = self.choose_threshold_dir_dialog(
            "Synchrony & connectivity",
            threshold_dirs,
            selected_dir=self.state.selected_connectivity_threshold_dir,
            label="Source spikes:",
        )
        if selected_source_dir is None:
            return None

        groups = self.ensure_non_aux_groups(session_manager, "Synchrony & connectivity")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Synchrony & connectivity")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        group_combo, channels_list = self.build_group_channel_selector(form, groups, header)

        save_checkbox = QCheckBox("Save outputs to folder")
        save_checkbox.setChecked(bool(self.state.output_folder))
        form.addRow("Output:", save_checkbox)
        plot_checkbox = QCheckBox("Plot image")
        plot_checkbox.setChecked(bool(self.state.raster_plot_show_image))
        form.addRow("", plot_checkbox)

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

        sttc_dt = QDoubleSpinBox()
        sttc_dt.setRange(0.1, 5000.0)
        sttc_dt.setDecimals(3)
        sttc_dt.setValue(float(self.state.sttc_dt_ms))
        form.addRow("STTC dt (ms):", sttc_dt)

        bin_ms = QDoubleSpinBox()
        bin_ms.setRange(0.1, 5000.0)
        bin_ms.setDecimals(3)
        bin_ms.setValue(float(self.state.sttc_bin_ms))
        form.addRow("Binning (ms):", bin_ms)

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
        if len(channels) < 2:
            QMessageBox.warning(dialog, "Synchrony & connectivity", "Select at least two channels.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Synchrony & connectivity", "Select output folder or disable saving.")
            return None
        self.state.output_folder = folder_edit.text().strip()
        self.state.sttc_dt_ms = float(sttc_dt.value())
        self.state.sttc_bin_ms = float(bin_ms.value())
        self.state.selected_connectivity_threshold_dir = str(selected_source_dir)
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=self.state.ignore_event_names,
            ignore_before_ms=self.state.ignore_before_ms,
            ignore_after_ms=self.state.ignore_after_ms,
        )
        return (
            selected_source_dir,
            channels,
            save_checkbox.isChecked(),
            plot_checkbox.isChecked(),
            folder_edit.text().strip(),
            float(sttc_dt.value()),
            float(bin_ms.value()),
        )

    @staticmethod
    def _within_dt_fraction(spikes_a: np.ndarray, spikes_b: np.ndarray, dt_s: float) -> float:
        if spikes_a.size == 0 or spikes_b.size == 0:
            return 0.0
        idx = np.searchsorted(spikes_b, spikes_a)
        left_ok = np.zeros(spikes_a.size, dtype=bool)
        right_ok = np.zeros(spikes_a.size, dtype=bool)
        mask_left = idx > 0
        left_ok[mask_left] = np.abs(spikes_a[mask_left] - spikes_b[idx[mask_left] - 1]) <= dt_s
        mask_right = idx < spikes_b.size
        right_ok[mask_right] = np.abs(spikes_b[idx[mask_right]] - spikes_a[mask_right]) <= dt_s
        return float(np.mean(left_ok | right_ok))

    @staticmethod
    def _time_covered_fraction(spikes: np.ndarray, dt_s: float, duration_s: float) -> float:
        if spikes.size == 0 or duration_s <= 0:
            return 0.0
        starts = np.maximum(0.0, spikes - dt_s)
        ends = np.minimum(duration_s, spikes + dt_s)
        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]
        total = 0.0
        cur_s = float(starts[0])
        cur_e = float(ends[0])
        for s, e in zip(starts[1:], ends[1:]):
            s = float(s)
            e = float(e)
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += max(0.0, cur_e - cur_s)
                cur_s = s
                cur_e = e
        total += max(0.0, cur_e - cur_s)
        return min(1.0, total / duration_s)

    def _sttc(self, spikes_a: np.ndarray, spikes_b: np.ndarray, dt_s: float, duration_s: float) -> float:
        pa = self._within_dt_fraction(spikes_a, spikes_b, dt_s)
        pb = self._within_dt_fraction(spikes_b, spikes_a, dt_s)
        ta = self._time_covered_fraction(spikes_a, dt_s, duration_s)
        tb = self._time_covered_fraction(spikes_b, dt_s, duration_s)
        part_a = (pa - tb) / (1.0 - pa * tb + 1e-12)
        part_b = (pb - ta) / (1.0 - pb * ta + 1e-12)
        return 0.5 * (part_a + part_b)

    @staticmethod
    def _binned_train(spikes_s: np.ndarray, duration_s: float, bin_s: float) -> np.ndarray:
        bins = max(1, int(np.ceil(duration_s / bin_s)))
        train = np.zeros(bins, dtype=np.float64)
        if spikes_s.size == 0:
            return train
        idx = np.floor(spikes_s / bin_s).astype(int)
        idx = idx[(0 <= idx) & (idx < bins)]
        if idx.size:
            np.add.at(train, idx, 1.0)
        return train

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        params = self._ask_parameters(session_manager, header, Path(add_on_data_dir))
        if params is None:
            return
        source_threshold_dir, channels, save_outputs, plot_image, output_folder, sttc_dt_ms, bin_ms = params
        sweep_idx = int(session_manager.gui_setup.current_sweep_idx)
        spikes_payload = self.read_spikes_payload(source_threshold_dir, sweep_idx)
        if spikes_payload is None:
            QMessageBox.warning(None, "Synchrony & connectivity", "No spikes payload for current sweep.")
            return

        times_by_channel = {}
        all_times = []
        for ch in channels:
            rows = spikes_payload.spikes_by_channel.get(int(ch), [])
            arr = np.asarray([float(row.time_ms) / 1000.0 for row in rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            arr.sort()
            times_by_channel[int(ch)] = arr
            if arr.size:
                all_times.append(arr)
        if all_times:
            duration_s = max(float(np.max(arr)) for arr in all_times) + 1e-6
        else:
            duration_s = max(1e-3, float(header.number_of_points_per_sweep[sweep_idx]) / float(header.sample_rate))

        n = len(channels)
        sttc_mat = np.eye(n, dtype=np.float64)
        corr_mat = np.eye(n, dtype=np.float64)
        dt_s = max(1e-6, float(sttc_dt_ms) / 1000.0)
        bin_s = max(1e-6, float(bin_ms) / 1000.0)
        binned = [self._binned_train(times_by_channel[int(ch)], duration_s, bin_s) for ch in channels]

        yield {"progress": 10, "message": "Computing pairwise synchrony..."}
        total_pairs = max(1, (n * (n - 1)) // 2)
        pair_i = 0
        for i in range(n):
            for j in range(i + 1, n):
                sttc = self._sttc(times_by_channel[int(channels[i])], times_by_channel[int(channels[j])], dt_s, duration_s)
                sttc_mat[i, j] = sttc_mat[j, i] = float(sttc)
                x = binned[i]
                y = binned[j]
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = float(np.corrcoef(x, y)[0, 1])
                else:
                    corr = 0.0
                corr_mat[i, j] = corr_mat[j, i] = corr
                pair_i += 1
                yield {"progress": 10 + int((pair_i / total_pairs) * 70), "message": f"Pair {pair_i}/{total_pairs}"}

        output_dir = Path(output_folder) if save_outputs else None
        canonical_dir = Path(add_on_data_dir) / source_threshold_dir.name
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_json = canonical_dir / f"{sweep_idx}.connectivity.json"
        canonical_payload = {
            "sweep_idx": sweep_idx,
            "source_threshold_dir": str(source_threshold_dir),
            "channels": [int(ch) for ch in channels],
            "sttc_dt_ms": float(sttc_dt_ms),
            "bin_ms": float(bin_ms),
            "sttc_matrix": sttc_mat.tolist(),
            "corr_matrix": corr_mat.tolist(),
        }
        canonical_json.write_text(json.dumps(canonical_payload, ensure_ascii=True, indent=2), encoding="utf-8")
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            base = output_dir / f"connectivity_{source_threshold_dir.name}_sw{sweep_idx}"
            np.save(str(base) + "_sttc.npy", sttc_mat)
            np.save(str(base) + "_corr.npy", corr_mat)
            np.savetxt(str(base) + "_sttc.csv", sttc_mat, delimiter=",")
            np.savetxt(str(base) + "_corr.csv", corr_mat, delimiter=",")

        labels = [self.channel_name(header, int(ch)) for ch in channels]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axes[0].imshow(sttc_mat, vmin=-1, vmax=1, cmap="coolwarm")
        axes[0].set_title("STTC")
        axes[0].set_xticks(range(n), labels, rotation=90)
        axes[0].set_yticks(range(n), labels)
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
        im1 = axes[1].imshow(corr_mat, vmin=-1, vmax=1, cmap="coolwarm")
        axes[1].set_title("Binned correlation")
        axes[1].set_xticks(range(n), labels, rotation=90)
        axes[1].set_yticks(range(n), labels)
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
        fig.tight_layout()
        if output_dir is not None:
            out_img = output_dir / f"connectivity_{safe_file_name(source_threshold_dir.name)}_sw{sweep_idx}.png"
            fig.savefig(str(out_img), dpi=200, bbox_inches="tight")
        if plot_image:
            plt.show()
        else:
            plt.close(fig)
        yield {"progress": 100, "message": "Synchrony & connectivity complete"}
