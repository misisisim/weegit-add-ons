from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
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

from weegit_add_ons.organoids_common import (
    FilterEditor,
    IgnoreEventsRule,
    NIRBaseAddOn,
    build_valid_mask,
    filter_from_spec,
    safe_file_name,
)


class SpectogramPlotAddOn(NIRBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    @staticmethod
    def _morlet_wavelet(scale: float, w: float, half_samples: int) -> np.ndarray:
        x = np.arange(-half_samples, half_samples + 1, dtype=np.float64)
        t = x / max(1e-12, float(scale))
        gauss = np.exp(-0.5 * (t ** 2))
        complex_sinus = np.exp(1j * w * t)
        wavelet = gauss * complex_sinus
        norm = np.sqrt(np.sum(np.abs(wavelet) ** 2))
        if norm > 0:
            wavelet = wavelet / norm
        return wavelet

    @staticmethod
    def _morlet_cwt_power(signal: np.ndarray, sample_rate: float, freqs: np.ndarray) -> np.ndarray:
        w = 6.0
        powers: List[np.ndarray] = []
        for freq in freqs:
            freq_hz = max(1e-6, float(freq))
            sigma = w / (2.0 * np.pi * freq_hz)
            half_window_s = max(3.5 * sigma, 2.0 / sample_rate)
            half_samples = max(8, int(round(half_window_s * sample_rate)))
            scale = (w * sample_rate) / (2.0 * np.pi * freq_hz)
            wavelet = SpectogramPlotAddOn._morlet_wavelet(scale=scale, w=w, half_samples=half_samples)
            conv = fftconvolve(signal, np.conjugate(wavelet[::-1]), mode="same")
            powers.append(np.abs(conv) ** 2)
        return np.asarray(powers, dtype=np.float64)

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
            Optional[dict],
            float,
            float,
            str,
        ]
    ]:
        groups = self.ensure_non_aux_groups(session_manager, "Spectogram plot")
        if groups is None:
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Spectogram plot")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

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
        plot_image_checkbox.setChecked(bool(self.state.spectrogram_show_image))
        form.addRow("", plot_image_checkbox)

        folder_edit = QLineEdit(self.state.output_folder or "")
        folder_edit.setEnabled(save_checkbox.isChecked())
        folder_btn = QPushButton("Browse")
        folder_btn.setEnabled(save_checkbox.isChecked())
        folder_row = QHBoxLayout()
        folder_row.addWidget(folder_edit, 1)
        folder_row.addWidget(folder_btn)
        form.addRow("Folder:", folder_row)

        filter_editor = FilterEditor(form, "Filtering", self.state.spectrogram_filter)

        transform_combo = QComboBox()
        transform_combo.addItems(["STFT", "Wavelet"])
        default_transform = (self.state.spectrogram_transform or "STFT").strip()
        default_idx = transform_combo.findText(default_transform)
        transform_combo.setCurrentIndex(max(0, default_idx))
        form.addRow("Transform:", transform_combo)

        freq_from_spin = QDoubleSpinBox()
        freq_from_spin.setRange(0.0, 1_000_000.0)
        freq_from_spin.setDecimals(3)
        freq_from_spin.setValue(float(self.state.spectrogram_freq_from_hz))
        form.addRow("Frequency from (Hz):", freq_from_spin)

        freq_to_spin = QDoubleSpinBox()
        freq_to_spin.setRange(0.0, 1_000_000.0)
        freq_to_spin.setDecimals(3)
        default_to = float(self.state.spectrogram_freq_to_hz)
        if default_to <= 0.0:
            default_to = 300.0
        freq_to_spin.setValue(default_to)
        form.addRow("Frequency to (Hz):", freq_to_spin)

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
            QMessageBox.warning(dialog, "Spectogram plot", "Select at least one channel.")
            return None
        if save_checkbox.isChecked() and not folder_edit.text().strip():
            QMessageBox.warning(dialog, "Spectogram plot", "Select output folder or disable saving.")
            return None

        ignore_names = self.selected_ignore_event_names(events_list)
        ignore_before = float(before_spin.value())
        ignore_after = float(after_spin.value())
        window_from_ms = float(window_from_spin.value())
        window_to_ms = float(window_to_spin.value())
        if window_to_ms <= window_from_ms:
            QMessageBox.warning(dialog, "Spectogram plot", "Window 'to' must be greater than 'from'.")
            return None
        freq_from_hz = float(freq_from_spin.value())
        freq_to_hz = float(freq_to_spin.value())
        nyquist = float(header.sample_rate) * 0.5
        if freq_to_hz <= freq_from_hz:
            QMessageBox.warning(dialog, "Spectogram plot", "Frequency 'to' must be greater than 'from'.")
            return None
        if freq_from_hz >= nyquist:
            QMessageBox.warning(
                dialog,
                "Spectogram plot",
                f"Frequency 'from' must be less than Nyquist ({nyquist:.3f} Hz).",
            )
            return None
        filter_spec = filter_editor.get_filter_spec()
        self.persist_common_selection(
            group_idx=int(group_combo.currentData()),
            channels=channels,
            ignore_event_names=ignore_names,
            ignore_before_ms=ignore_before,
            ignore_after_ms=ignore_after,
        )
        self.state.output_folder = folder_edit.text().strip()
        self.state.spectrogram_filter = filter_spec
        self.state.spectrogram_freq_from_hz = freq_from_hz
        self.state.spectrogram_freq_to_hz = freq_to_hz
        self.state.spectrogram_transform = transform_combo.currentText().strip()
        self.state.spectrogram_show_image = plot_image_checkbox.isChecked()
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
            filter_spec,
            freq_from_hz,
            freq_to_hz,
            transform_combo.currentText().strip(),
        )

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
            filter_spec,
            freq_from_hz,
            freq_to_hz,
            transform_name,
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

        selected_filter = filter_from_spec(filter_spec)
        nyquist = sample_rate * 0.5
        f_from = max(0.0, float(freq_from_hz))
        f_to = min(float(freq_to_hz), nyquist)
        output_dir = Path(output_folder) if save_outputs else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)

        total = len(channel_indexes)
        yield {"progress": 0, "message": "Generating spectograms..."}
        for idx, channel_idx in enumerate(channel_indexes, start=1):
            channel_name = self.channel_name(header, int(channel_idx))
            signal = session_manager.experiment_data.process_single_channel(
                channel_idx=int(channel_idx),
                sweep_idx=sweep_idx,
                start_sample=start_sample,
                end_sample=end_sample,
                each_point=1,
                sample_rate=sample_rate,
                filters=[selected_filter] if selected_filter is not None else [],
                output_number_of_dots=n_samples,
                transformation_add_ons=[],
            )
            signal = np.asarray(signal, dtype=np.float64)
            if valid_mask.size == signal.size:
                signal[~valid_mask] = 0.0

            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            if transform_name == "Wavelet":
                # Morlet-based CWT scalogram.
                n_freqs = 128
                freqs = np.linspace(max(1e-6, f_from), max(f_from + 1e-6, f_to), n_freqs)
                power = self._morlet_cwt_power(signal=signal, sample_rate=sample_rate, freqs=freqs)
                ax.imshow(
                    power,
                    extent=[start_second, end_second, float(freqs[0]), float(freqs[-1])],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                )
            else:
                nfft = min(1024, max(64, signal.size // 8))
                noverlap = max(0, int(nfft * 0.5))
                _, _, _, _ = ax.specgram(
                    signal,
                    NFFT=nfft,
                    Fs=sample_rate,
                    noverlap=noverlap,
                    cmap="viridis",
                )
            ax.set_xlim(start_second, end_second)
            ax.set_ylim(f_from, f_to)
            ax.set_title(f"Spectogram ({transform_name}) | {channel_name} | sweep {sweep_idx}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")

            if output_dir is not None:
                fig.savefig(
                    str(output_dir / f"spectogram_{safe_file_name(channel_name)}_sw{sweep_idx}.png"),
                    dpi=200,
                    bbox_inches="tight",
                )
            if plot_image:
                plt.show()
            else:
                plt.close(fig)
            yield {"progress": int((idx / total) * 100), "message": f"Plotted channel {idx}/{total}"}
