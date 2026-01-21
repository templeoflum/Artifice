"""
Inspector panel for editing node parameters.

Displays and allows editing of parameters for the selected node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QSlider,
    QScrollArea,
    QFrame,
    QGroupBox,
    QSizePolicy,
)

if TYPE_CHECKING:
    from artifice.core.node import Node, ParameterType


class ParameterWidget(QWidget):
    """
    Base class for parameter editor widgets.

    Subclasses implement specific parameter types.
    """

    value_changed = Signal(str, object)  # param_name, value

    def __init__(self, name: str, param_info: dict, parent=None):
        super().__init__(parent)

        self._name = name
        self._param_info = param_info

        self._setup_ui()

    @property
    def name(self) -> str:
        """Get the parameter name."""
        return self._name

    def _setup_ui(self) -> None:
        """Set up the widget. Override in subclasses."""
        pass

    def set_value(self, value: Any) -> None:
        """Set the widget value. Override in subclasses."""
        pass

    def get_value(self) -> Any:
        """Get the widget value. Override in subclasses."""
        pass


class IntParameterWidget(ParameterWidget):
    """Widget for integer parameters."""

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel(self._name.replace("_", " ").title())
        label.setFixedWidth(100)
        layout.addWidget(label)

        # Slider (if range specified)
        min_val = self._param_info.get("min_value", 0)
        max_val = self._param_info.get("max_value", 100)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(min_val, max_val)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        # Spinbox
        self._spinbox = QSpinBox()
        self._spinbox.setRange(min_val, max_val)
        self._spinbox.setFixedWidth(60)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self._spinbox)

    def _on_slider_changed(self, value: int) -> None:
        self._spinbox.blockSignals(True)
        self._spinbox.setValue(value)
        self._spinbox.blockSignals(False)
        self.value_changed.emit(self._name, value)

    def _on_spinbox_changed(self, value: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(value)
        self._slider.blockSignals(False)
        self.value_changed.emit(self._name, value)

    def set_value(self, value: int) -> None:
        self._slider.blockSignals(True)
        self._spinbox.blockSignals(True)
        self._slider.setValue(value)
        self._spinbox.setValue(value)
        self._slider.blockSignals(False)
        self._spinbox.blockSignals(False)

    def get_value(self) -> int:
        return self._spinbox.value()


class FloatParameterWidget(ParameterWidget):
    """Widget for float parameters."""

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel(self._name.replace("_", " ").title())
        label.setFixedWidth(100)
        layout.addWidget(label)

        # Slider
        self._min_val = self._param_info.get("min_value", 0.0)
        self._max_val = self._param_info.get("max_value", 1.0)
        step = self._param_info.get("step", 0.01)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)  # Map to 0-1000 range internally
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, 1)

        # Spinbox
        self._spinbox = QDoubleSpinBox()
        self._spinbox.setRange(self._min_val, self._max_val)
        self._spinbox.setSingleStep(step)
        self._spinbox.setDecimals(3)
        self._spinbox.setFixedWidth(70)
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self._spinbox)

    def _value_to_slider(self, value: float) -> int:
        """Convert float value to slider position."""
        if self._max_val == self._min_val:
            return 0
        normalized = (value - self._min_val) / (self._max_val - self._min_val)
        return int(normalized * 1000)

    def _slider_to_value(self, pos: int) -> float:
        """Convert slider position to float value."""
        normalized = pos / 1000.0
        return self._min_val + normalized * (self._max_val - self._min_val)

    def _on_slider_changed(self, pos: int) -> None:
        value = self._slider_to_value(pos)
        self._spinbox.blockSignals(True)
        self._spinbox.setValue(value)
        self._spinbox.blockSignals(False)
        self.value_changed.emit(self._name, value)

    def _on_spinbox_changed(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(self._value_to_slider(value))
        self._slider.blockSignals(False)
        self.value_changed.emit(self._name, value)

    def set_value(self, value: float) -> None:
        self._slider.blockSignals(True)
        self._spinbox.blockSignals(True)
        self._slider.setValue(self._value_to_slider(value))
        self._spinbox.setValue(value)
        self._slider.blockSignals(False)
        self._spinbox.blockSignals(False)

    def get_value(self) -> float:
        return self._spinbox.value()


class BoolParameterWidget(ParameterWidget):
    """Widget for boolean parameters."""

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._checkbox = QCheckBox(self._name.replace("_", " ").title())
        self._checkbox.toggled.connect(self._on_toggled)
        layout.addWidget(self._checkbox)
        layout.addStretch()

    def _on_toggled(self, checked: bool) -> None:
        self.value_changed.emit(self._name, checked)

    def set_value(self, value: bool) -> None:
        self._checkbox.blockSignals(True)
        self._checkbox.setChecked(value)
        self._checkbox.blockSignals(False)

    def get_value(self) -> bool:
        return self._checkbox.isChecked()


class EnumParameterWidget(ParameterWidget):
    """Widget for enum/choice parameters."""

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel(self._name.replace("_", " ").title())
        label.setFixedWidth(100)
        layout.addWidget(label)

        # Combo box
        self._combo = QComboBox()
        choices = self._param_info.get("choices", [])
        self._combo.addItems([str(c) for c in choices])
        self._combo.currentTextChanged.connect(self._on_changed)
        layout.addWidget(self._combo, 1)

    def _on_changed(self, text: str) -> None:
        self.value_changed.emit(self._name, text)

    def set_value(self, value: str) -> None:
        self._combo.blockSignals(True)
        index = self._combo.findText(str(value))
        if index >= 0:
            self._combo.setCurrentIndex(index)
        self._combo.blockSignals(False)

    def get_value(self) -> str:
        return self._combo.currentText()


class StringParameterWidget(ParameterWidget):
    """Widget for string parameters."""

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        label = QLabel(self._name.replace("_", " ").title())
        label.setFixedWidth(100)
        layout.addWidget(label)

        # Line edit
        self._edit = QLineEdit()
        self._edit.editingFinished.connect(self._on_changed)
        layout.addWidget(self._edit, 1)

    def _on_changed(self) -> None:
        self.value_changed.emit(self._name, self._edit.text())

    def set_value(self, value: str) -> None:
        self._edit.blockSignals(True)
        self._edit.setText(str(value))
        self._edit.blockSignals(False)

    def get_value(self) -> str:
        return self._edit.text()


class InspectorPanel(QWidget):
    """
    Panel for inspecting and editing node parameters.

    Shows the selected node's parameters with appropriate editor widgets.
    """

    parameter_changed = Signal(object, str, object)  # node, param_name, value

    def __init__(self, parent=None):
        super().__init__(parent)

        self._node: Node | None = None
        self._param_widgets: dict[str, ParameterWidget] = {}

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        self._header = QLabel("No selection")
        self._header.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._header)

        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameStyle(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Container for parameter widgets
        self._param_container = QWidget()
        self._param_layout = QVBoxLayout(self._param_container)
        self._param_layout.setContentsMargins(0, 0, 0, 0)
        self._param_layout.setSpacing(8)
        self._param_layout.addStretch()

        scroll.setWidget(self._param_container)
        layout.addWidget(scroll)

    def set_node(self, node: Node) -> None:
        """Set the node to inspect."""
        self._node = node
        self._update_ui()

    def clear(self) -> None:
        """Clear the inspector."""
        self._node = None
        self._header.setText("No selection")

        # Remove all parameter widgets
        for widget in self._param_widgets.values():
            widget.deleteLater()
        self._param_widgets.clear()

    def _update_ui(self) -> None:
        """Update the UI to show current node's parameters."""
        # Clear existing widgets
        for widget in self._param_widgets.values():
            self._param_layout.removeWidget(widget)
            widget.deleteLater()
        self._param_widgets.clear()

        if not self._node:
            self._header.setText("No selection")
            return

        # Update header
        self._header.setText(f"{self._node.name}")

        # Create widgets for each parameter
        for name, param in self._node.parameters.items():
            widget = self._create_param_widget(name, param)
            if widget:
                # Insert before the stretch
                self._param_layout.insertWidget(
                    self._param_layout.count() - 1, widget
                )
                self._param_widgets[name] = widget

                # Connect signal
                widget.value_changed.connect(self._on_param_changed)

                # Set initial value
                widget.set_value(param.value)

    def _create_param_widget(self, name: str, param) -> ParameterWidget | None:
        """Create appropriate widget for parameter type."""
        param_info = {
            "min_value": getattr(param, "min_value", None),
            "max_value": getattr(param, "max_value", None),
            "step": getattr(param, "step", None),
            "choices": getattr(param, "choices", None),
        }

        param_type = param.param_type.name if hasattr(param.param_type, "name") else str(param.param_type)

        if param_type == "INT":
            return IntParameterWidget(name, param_info)
        elif param_type == "FLOAT":
            return FloatParameterWidget(name, param_info)
        elif param_type == "BOOL":
            return BoolParameterWidget(name, param_info)
        elif param_type == "ENUM":
            return EnumParameterWidget(name, param_info)
        elif param_type == "STRING" or param_type == "PATH":
            return StringParameterWidget(name, param_info)
        else:
            # Default to string for unknown types
            return StringParameterWidget(name, param_info)

    def _on_param_changed(self, name: str, value: Any) -> None:
        """Handle parameter value change."""
        if self._node:
            # Update the node parameter
            try:
                self._node.set_parameter(name, value)
                self.parameter_changed.emit(self._node, name, value)
            except Exception as e:
                print(f"Error setting parameter {name}: {e}")

    def find_widget(self, param_name: str) -> ParameterWidget | None:
        """Find a parameter widget by name."""
        return self._param_widgets.get(param_name)
