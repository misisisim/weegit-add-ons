from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
from scipy.signal import find_peaks

from weegit.core.add_ons.base import BaseAddOn
from weegit.core.conversions.filters import (
    all_filter_names,
    filter_class_by_name,
    ButterworthLowPassFilter,
    ButterworthHighPassFilter,
    ButterworthBandPassFilter,
    ChebyshevBandPassFilter,
    NotchFilter,
    BaseFilter,
)


class DigitalEventsAddOn(BaseAddOn):
    """
    Detects digital events on selected EEG channels and appends them
    to UserSession.events using a newly created vocabulary name.
    """

    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    def _ask_parameters(
        self, session_manager, header
    ) -> Optional[Tuple[List[int], str, float, float, Optional[BaseFilter]]]:
        user_session = session_manager.user_session
        digital_channels: List[int] = list(user_session.eeg_channel_indexes or [])
        if not digital_channels:
            QMessageBox.warning(None, "Digital events", "No EEG channels configured.")
            return None

        dialog = QDialog()
        dialog.setWindowTitle("Digital events detection")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()

        # Channel multi-selector (all selected by default)
        channels_list = QListWidget()
        channels_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for ch in digital_channels:
            name = ""
            try:
                if 0 <= ch < len(header.channel_info.name):
                    name = header.channel_info.name[ch]
            except Exception:
                name = ""
            label = f"{ch}" if not name else f"{ch} [{name}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ch)
            channels_list.addItem(item)
            item.setSelected(True)
        form.addRow("Digital channels:", channels_list)

        # Event name
        name_edit = QLineEdit()
        form.addRow("Event name:", name_edit)

        # Height (can be negative; no fixed default requested)
        height_spin = QDoubleSpinBox()
        height_spin.setDecimals(6)
        height_spin.setRange(-1e9, 1e9)
        form.addRow("Height (threshold):", height_spin)
        form.addRow(
            "",
            QLabel(
                "Tip: for downward events use a negative threshold "
                "(e.g. -200 means detect values below -200 uV)."
            ),
        )

        # Minimal distance between events in ms
        distance_spin = QDoubleSpinBox()
        distance_spin.setDecimals(3)
        distance_spin.setRange(0.0, 1e9)
        distance_spin.setValue(500.0)
        form.addRow("Min distance, ms:", distance_spin)

        # Optional single filter (independent from gui_setup.filters)
        use_filter_checkbox = QCheckBox("Use filter")
        use_filter_checkbox.setChecked(False)
        form.addRow("Filtering:", use_filter_checkbox)

        filter_selector = QComboBox()
        filter_selector.addItems(all_filter_names())
        filter_selector.setEnabled(False)
        form.addRow("Filter type:", filter_selector)

        filter_params_container = QWidget()
        filter_params_widget = QVBoxLayout(filter_params_container)
        filter_params_widget.setContentsMargins(0, 0, 0, 0)
        form.addRow("Filter params:", filter_params_container)

        filter_param_inputs: Dict[str, object] = {}

        def _clear_layout_items(layout_obj) -> None:
            while layout_obj.count():
                item = layout_obj.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.deleteLater()
                elif child_layout is not None:
                    _clear_layout_items(child_layout)

        def _clear_filter_params() -> None:
            _clear_layout_items(filter_params_widget)
            filter_param_inputs.clear()

        def _add_param_spin(label: str, key: str, value: float, min_v: float, max_v: float, step: float) -> None:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QDoubleSpinBox()
            spin.setRange(min_v, max_v)
            spin.setSingleStep(step)
            spin.setDecimals(6)
            spin.setValue(float(value))
            row.addWidget(spin)
            filter_params_widget.addLayout(row)
            filter_param_inputs[key] = spin

        def _add_param_int(label: str, key: str, value: int, min_v: int, max_v: int) -> None:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin = QSpinBox()
            spin.setRange(min_v, max_v)
            spin.setSingleStep(1)
            spin.setValue(int(value))
            row.addWidget(spin)
            filter_params_widget.addLayout(row)
            filter_param_inputs[key] = spin

        def _rebuild_filter_params() -> None:
            _clear_filter_params()
            if not use_filter_checkbox.isChecked():
                return
            selected_name = filter_selector.currentText().strip()
            cls = filter_class_by_name(selected_name)
            if cls is None:
                return
            flt = cls()
            if isinstance(flt, ButterworthLowPassFilter):
                _add_param_spin("Cutoff (Hz):", "cutoff_hz", flt.cutoff_hz, 0.1, 1e6, 1.0)
                _add_param_int("Order:", "order", flt.order, 1, 12)
            elif isinstance(flt, ButterworthHighPassFilter):
                _add_param_spin("Cutoff (Hz):", "cutoff_hz", flt.cutoff_hz, 0.1, 1e6, 1.0)
                _add_param_int("Order:", "order", flt.order, 1, 12)
            elif isinstance(flt, ButterworthBandPassFilter):
                _add_param_spin("Low cut (Hz):", "lowcut_hz", flt.lowcut_hz, 0.1, 1e6, 1.0)
                _add_param_spin("High cut (Hz):", "highcut_hz", flt.highcut_hz, 0.1, 1e6, 1.0)
                _add_param_int("Order:", "order", flt.order, 1, 12)
            elif isinstance(flt, ChebyshevBandPassFilter):
                _add_param_spin("Low cut (Hz):", "lowcut_hz", flt.lowcut_hz, 0.1, 1e6, 1.0)
                _add_param_spin("High cut (Hz):", "highcut_hz", flt.highcut_hz, 0.1, 1e6, 1.0)
                _add_param_int("Order:", "order", flt.order, 1, 12)
                _add_param_spin("Ripple (dB):", "ripple_db", flt.ripple_db, 0.1, 10.0, 0.1)
            elif isinstance(flt, NotchFilter):
                _add_param_spin("Notch (Hz):", "notch_freq_hz", flt.notch_freq_hz, 1.0, 1e6, 1.0)
                _add_param_spin("Q factor:", "q_factor", flt.q_factor, 0.1, 200.0, 1.0)

        def _on_use_filter_changed(_state: int) -> None:
            enabled = use_filter_checkbox.isChecked()
            filter_selector.setEnabled(enabled)
            _rebuild_filter_params()

        use_filter_checkbox.stateChanged.connect(_on_use_filter_changed)
        filter_selector.currentIndexChanged.connect(lambda _idx: _rebuild_filter_params())
        _rebuild_filter_params()

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

        event_name = name_edit.text().strip()
        if not event_name:
            QMessageBox.warning(dialog, "Digital events", "Event name must not be empty.")
            return None

        # Ensure uniqueness vs existing vocabulary
        if event_name in (session_manager.events_vocabulary or {}).values():
            QMessageBox.warning(dialog, "Digital events", "Event name must be unique.")
            return None

        selected_items = channels_list.selectedItems()
        selected_channels: List[int] = []
        for item in selected_items:
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is not None:
                selected_channels.append(int(value))
        if not selected_channels:
            QMessageBox.warning(dialog, "Digital events", "Select at least one channel.")
            return None

        height = float(height_spin.value())
        distance_ms = float(distance_spin.value())
        selected_filter: Optional[BaseFilter] = None
        if use_filter_checkbox.isChecked():
            selected_name = filter_selector.currentText().strip()
            cls = filter_class_by_name(selected_name)
            if cls is None:
                QMessageBox.warning(dialog, "Digital events", "Unknown filter selected.")
                return None
            flt = cls()
            setattr(flt, "enabled", True)
            for key, widget in filter_param_inputs.items():
                if isinstance(widget, QDoubleSpinBox):
                    raw_value = float(widget.value())
                    current = getattr(flt, key, raw_value)
                    if isinstance(current, int):
                        setattr(flt, key, int(raw_value))
                    else:
                        setattr(flt, key, raw_value)
                elif isinstance(widget, QSpinBox):
                    setattr(flt, key, int(widget.value()))
            if hasattr(flt, "sos_cache"):
                flt.sos_cache = {}
            selected_filter = flt
        return selected_channels, event_name, height, distance_ms, selected_filter

    def run(self, session_manager, add_on_data_dir):
        header = session_manager.header
        experiment_data = session_manager.experiment_data

        params = self._ask_parameters(session_manager, header)
        if params is None:
            return

        channel_indexes, event_name, height, distance_ms, selected_filter = params

        add_on_data_dir = Path(add_on_data_dir) if add_on_data_dir is not None else None
        if add_on_data_dir is not None:
            add_on_data_dir.mkdir(parents=True, exist_ok=True)

        total_sweeps = header.number_of_sweeps
        if total_sweeps <= 0:
            yield {"progress": 100, "message": "No sweeps to process"}
            return

        yield {"progress": 0, "message": "Detecting digital events..."}

        # Convert distance from ms to samples
        distance_samples = int((distance_ms * header.sample_rate) / 1000.0) if distance_ms > 0 else 1
        distance_samples = max(distance_samples, 1)
        one_filter: List[BaseFilter] = [selected_filter] if selected_filter is not None else []
        detected_events_by_sweep: List[Tuple[int, float]] = []

        for sweep_idx in range(total_sweeps):
            found_in_sweep = 0
            for channel_idx in channel_indexes:
                # Full sweep, filtered
                start_sample = 0
                end_sample = header.number_of_points_per_sweep
                output_number_of_dots = header.number_of_points_per_sweep
                each_point = 1

                signal = experiment_data.process_single_channel(
                    channel_idx=channel_idx,
                    sweep_idx=sweep_idx,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    each_point=each_point,
                    sample_rate=header.sample_rate,
                    filters=one_filter,
                    output_number_of_dots=output_number_of_dots,
                    apply_filters=True,
                    transformation_add_ons=[],
                )
                signal = np.asarray(signal).reshape(-1)

                # If height < 0, invert signal so that peaks become positive
                search_signal = signal.copy()
                search_height = height
                if height < 0:
                    search_signal = -search_signal
                    search_height = -height

                peaks, _ = find_peaks(search_signal, height=search_height, distance=distance_samples)
                for peak_sample in peaks:
                    time_ms = (peak_sample * 1000.0) / header.sample_rate
                    detected_events_by_sweep.append((sweep_idx, float(time_ms)))
                found_in_sweep += int(peaks.size)

            progress = int(((sweep_idx + 1) / total_sweeps) * 100)
            message = (
                f"Sweep {sweep_idx + 1}/{total_sweeps}: no events"
                if found_in_sweep == 0
                else f"Sweep {sweep_idx + 1}/{total_sweeps} processed"
            )
            yield {
                "progress": progress,
                "message": message,
            }

        if not detected_events_by_sweep:
            QMessageBox.information(
                None,
                "Digital events",
                "No events were detected. Event vocabulary was not created.",
            )
            return

        event_name_id = session_manager.add_event_vocabulary(event_name)
        events_specs: List[Tuple[int, int, float]] = [
            (event_name_id, sweep_idx, time_ms)
            for sweep_idx, time_ms in detected_events_by_sweep
        ]
        session_manager.add_events(events_specs)
