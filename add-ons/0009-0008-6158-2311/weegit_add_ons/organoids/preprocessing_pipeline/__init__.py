from __future__ import annotations

from typing import Dict, Optional

from PyQt6.QtWidgets import (
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
)
from weegit.core.add_ons.base import BaseAddOn

from weegit_add_ons.organoids.common import OrganoidsBaseAddOn
from weegit_add_ons.organoids.preprocessing import (
    PipelineSpec,
    PreprocessingStep,
    read_pipeline_store,
    write_pipeline_store,
)


STEP_LABELS: Dict[str, str] = {
    "notch": "Notch",
    "highpass": "High-pass",
    "lowpass": "Low-pass",
    "bandpass": "Band-pass",
    "cmr": "Common median/mean reference",
    "artifact_removal": "Artifact removal",
}


class PreprocessingPipelineAddOn(OrganoidsBaseAddOn, BaseAddOn):
    TRANSFORMATION = False
    VIEWABLE = False
    RUNNABLE = True

    @staticmethod
    def _step_label(step: PreprocessingStep) -> str:
        params = ", ".join(f"{k}={v}" for k, v in step.params.items())
        prefix = STEP_LABELS.get(step.kind, step.kind)
        return f"{prefix} ({params})" if params else prefix

    @staticmethod
    def _ask_step(dialog_parent) -> Optional[PreprocessingStep]:
        dialog = QDialog(dialog_parent)
        dialog.setWindowTitle("Add preprocessing step")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        kind_combo = QComboBox()
        for key, label in STEP_LABELS.items():
            kind_combo.addItem(label, key)
        form.addRow("Step:", kind_combo)

        params_layout = QFormLayout()
        form.addRow(QLabel("Parameters:"))
        form.addRow(params_layout)
        widgets = {}

        def clear_params() -> None:
            while params_layout.count():
                item = params_layout.takeAt(0)
                widget = item.widget()
                layout_obj = item.layout()
                if widget is not None:
                    widget.deleteLater()
                elif layout_obj is not None:
                    while layout_obj.count():
                        child = layout_obj.takeAt(0).widget()
                        if child is not None:
                            child.deleteLater()
            widgets.clear()

        def add_double(label: str, key: str, value: float, low: float = 0.0, high: float = 1_000_000.0) -> None:
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(6)
            spin.setValue(float(value))
            params_layout.addRow(label, spin)
            widgets[key] = spin

        def add_int(label: str, key: str, value: int, low: int = 1, high: int = 32) -> None:
            spin = QSpinBox()
            spin.setRange(low, high)
            spin.setValue(int(value))
            params_layout.addRow(label, spin)
            widgets[key] = spin

        def add_combo(label: str, key: str, values: list[str], current: str) -> None:
            combo = QComboBox()
            combo.addItems(values)
            idx = combo.findText(current)
            combo.setCurrentIndex(max(0, idx))
            params_layout.addRow(label, combo)
            widgets[key] = combo

        def rebuild_params() -> None:
            clear_params()
            kind = str(kind_combo.currentData())
            if kind == "notch":
                add_double("Frequency (Hz):", "notch_freq_hz", 50.0, 1.0)
                add_double("Q factor:", "q_factor", 30.0, 0.1, 300.0)
            elif kind == "highpass":
                add_double("Cutoff (Hz):", "cutoff_hz", 300.0, 0.1)
                add_int("Order:", "order", 3, 1, 12)
            elif kind == "lowpass":
                add_double("Cutoff (Hz):", "cutoff_hz", 3000.0, 0.1)
                add_int("Order:", "order", 3, 1, 12)
            elif kind == "bandpass":
                add_double("Low cut (Hz):", "lowcut_hz", 300.0, 0.1)
                add_double("High cut (Hz):", "highcut_hz", 3000.0, 0.1)
                add_int("Order:", "order", 3, 1, 12)
            elif kind == "cmr":
                add_combo("Method:", "method", ["median", "mean"], "median")
            elif kind == "artifact_removal":
                add_double("Threshold (robust z):", "threshold_z", 5.0, 0.1, 1000.0)
                add_double("Min distance (ms):", "min_distance_ms", 20.0, 0.0, 60_000.0)
                add_double("Pre window (ms):", "pre_ms", 5.0, 0.0, 60_000.0)
                add_double("Post window (ms):", "post_ms", 25.0, 0.0, 60_000.0)
                add_double("Merge gap (ms):", "merge_gap_ms", 5.0, 0.0, 60_000.0)

        kind_combo.currentIndexChanged.connect(lambda _idx: rebuild_params())
        rebuild_params()

        layout.addLayout(form)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("Add")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_ok)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_ok.clicked.connect(dialog.accept)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        params = {}
        for key, widget in widgets.items():
            if isinstance(widget, QComboBox):
                params[key] = widget.currentText().strip()
            else:
                params[key] = widget.value()
        return PreprocessingStep(kind=str(kind_combo.currentData()), enabled=True, params=params)

    def run(self, session_manager, add_on_data_dir):
        store = read_pipeline_store(self.pipelines_path(add_on_data_dir))
        dialog = QDialog()
        dialog.setWindowTitle("Preprocessing pipeline")
        dialog.setMinimumWidth(620)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        name_edit = QLineEdit(self.state.selected_pipeline_name if self.state.selected_pipeline_name != "raw" else "")
        form.addRow("Pipeline name:", name_edit)
        description_edit = QLineEdit()
        form.addRow("Description:", description_edit)

        steps_list = QListWidget()
        form.addRow("Ordered steps:", steps_list)

        steps: list[PreprocessingStep] = []

        def refresh() -> None:
            steps_list.clear()
            for idx, step in enumerate(steps, start=1):
                item = QListWidgetItem(f"{idx}. {self._step_label(step)}")
                steps_list.addItem(item)

        def add_step() -> None:
            step = self._ask_step(dialog)
            if step is not None:
                steps.append(step)
                refresh()

        def remove_step() -> None:
            row = steps_list.currentRow()
            if 0 <= row < len(steps):
                steps.pop(row)
                refresh()

        def move_step(delta: int) -> None:
            row = steps_list.currentRow()
            new_row = row + delta
            if 0 <= row < len(steps) and 0 <= new_row < len(steps):
                steps[row], steps[new_row] = steps[new_row], steps[row]
                refresh()
                steps_list.setCurrentRow(new_row)

        buttons = QHBoxLayout()
        btn_add = QPushButton("Add step")
        btn_remove = QPushButton("Remove")
        btn_up = QPushButton("Up")
        btn_down = QPushButton("Down")
        buttons.addWidget(btn_add)
        buttons.addWidget(btn_remove)
        buttons.addWidget(btn_up)
        buttons.addWidget(btn_down)
        form.addRow("", buttons)
        btn_add.clicked.connect(add_step)
        btn_remove.clicked.connect(remove_step)
        btn_up.clicked.connect(lambda: move_step(-1))
        btn_down.clicked.connect(lambda: move_step(1))

        layout.addLayout(form)
        actions = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_save = QPushButton("Save")
        actions.addStretch(1)
        actions.addWidget(btn_cancel)
        actions.addWidget(btn_save)
        layout.addLayout(actions)
        btn_cancel.clicked.connect(dialog.reject)
        btn_save.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name = name_edit.text().strip()
        if not name:
            QMessageBox.warning(dialog, "Preprocessing pipeline", "Pipeline name is required.")
            return
        if name == "raw":
            QMessageBox.warning(dialog, "Preprocessing pipeline", "'raw' is reserved for an empty pipeline.")
            return
        spec = PipelineSpec(name=name, description=description_edit.text().strip(), steps=steps)
        store[name] = spec
        path = write_pipeline_store(self.pipelines_path(add_on_data_dir), store)
        self.state.selected_pipeline_name = name
        yield {"progress": 100, "message": f"Saved preprocessing pipeline '{name}' to {path.name}"}
