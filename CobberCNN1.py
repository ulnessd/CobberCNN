import sys
import os
import numpy as np
from typing import Dict, Tuple

# --- Matplotlib and PyQt6 Integration ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QComboBox, QListWidget, QPushButton, QTabWidget,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# --- Core Engine ---

IMAGES: Dict[str, np.ndarray] = {
    "Propane": np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    "Isobutane": np.array([  # T-shape
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]),
    "Neopentane": np.array([  # + shape
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
}
FILTERS: Dict[str, np.ndarray] = {
    "Vertical Line": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]),
    "Horizontal Line": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]),
    "Junction Detector": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "Diagonal (45°)": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
}


def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image_h, image_w = image.shape;
    kernel_h, kernel_w = kernel.shape
    output_h, output_w = image_h - kernel_h + 1, image_w - kernel_w + 1
    feature_map = np.zeros((output_h, output_w))
    for y in range(output_h):
        for x in range(output_w):
            receptive_field = image[y:y + kernel_h, x:x + kernel_w]
            feature_map[y, x] = np.sum(receptive_field * kernel)
    return feature_map


def relu(feature_map: np.ndarray) -> np.ndarray: return np.maximum(0, feature_map)


# --- GUI Development ---

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberCNN: A Convolutional Neural Network Explorer")
        self.setGeometry(100, 100, 1400, 600)
        self.current_input_image = IMAGES["Propane"]
        self.feature_maps: Dict[str, np.ndarray] = {}
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)

        main_layout = QHBoxLayout()
        controls_panel = QFrame();
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)
        process_panel = QFrame();
        process_panel.setFrameShape(QFrame.Shape.StyledPanel)
        process_layout = QVBoxLayout(process_panel)
        self.output_tabs = QTabWidget()

        controls_layout.addWidget(QLabel("<h3>1. Input Selection</h3>"))
        self.molecule_selector = QComboBox();
        self.molecule_selector.addItems(IMAGES.keys())
        controls_layout.addWidget(self.molecule_selector)
        self.input_image_canvas = MplCanvas(self, width=2, height=2, dpi=70)
        controls_layout.addWidget(self.input_image_canvas)
        controls_layout.addWidget(QLabel("<h3>2. Filter Library</h3>"))
        self.filter_list = QListWidget();
        self.filter_list.addItems(FILTERS.keys())
        controls_layout.addWidget(self.filter_list)
        self.apply_to_input_button = QPushButton("Apply to Input Image")
        self.apply_to_feature_map_button = QPushButton("Apply to Current Feature Map")
        controls_layout.addWidget(self.apply_to_input_button)
        controls_layout.addWidget(self.apply_to_feature_map_button)
        controls_layout.addStretch()

        self.save_button = QPushButton("Save All Open Maps")
        self.clear_button = QPushButton("Clear/Reset All")
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.clear_button)

        process_layout.addWidget(QLabel("<h3>Convolution Process</h3>"))
        self.process_canvas = MplCanvas(self)
        process_layout.addWidget(self.process_canvas)

        main_layout.addWidget(controls_panel, 2);
        main_layout.addWidget(process_panel, 3)
        main_layout.addWidget(self.output_tabs, 3)

        central_widget = QWidget();
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.molecule_selector.currentTextChanged.connect(self.update_input_image)
        self.apply_to_input_button.clicked.connect(self.start_convolution_on_input)
        self.apply_to_feature_map_button.clicked.connect(self.start_convolution_on_feature_map)
        self.save_button.clicked.connect(self.save_all_maps)
        self.clear_button.clicked.connect(self.clear_all_tabs)
        self.update_input_image("Propane")

    def update_input_image(self, name):
        self.current_input_image = IMAGES[name]
        self.draw_image(self.input_image_canvas, self.current_input_image, f"Input: {name}")
        self.draw_image(self.process_canvas, self.current_input_image, "Ready to Convolve")

    def start_convolution_on_input(self):
        self.start_animation(self.current_input_image, self.molecule_selector.currentText())

    def start_convolution_on_feature_map(self):
        current_tab_index = self.output_tabs.currentIndex()
        if current_tab_index < 0:
            QMessageBox.warning(self, "No Feature Map", "Please generate a feature map first.");
            return
        tab_name = self.output_tabs.tabText(current_tab_index)
        input_feature_map = self.feature_maps.get(tab_name)
        if input_feature_map is None: return
        self.start_animation(input_feature_map, tab_name)

    def start_animation(self, input_image, base_name):
        if not self.filter_list.currentItem():
            QMessageBox.warning(self, "No Filter Selected", "Please select a filter from the library.");
            return

        self.set_buttons_enabled(False)
        self.anim_input_image = input_image
        self.anim_kernel = FILTERS[self.filter_list.currentItem().text()]
        self.anim_pos = (0, 0)
        output_h, output_w = input_image.shape[0] - 2, input_image.shape[1] - 2
        self.anim_feature_map = np.zeros((output_h, output_w))

        new_tab_name = f"{base_name} + {self.filter_list.currentItem().text()}"

        self.current_feature_map_canvas = MplCanvas(self)
        self.output_tabs.addTab(self.current_feature_map_canvas, new_tab_name)
        self.output_tabs.setCurrentWidget(self.current_feature_map_canvas)

        self.draw_image(self.process_canvas, self.anim_input_image, "Convolving...")
        self.animation_timer.start(30)

    def animation_step(self):
        y, x = self.anim_pos
        receptive_field = self.anim_input_image[y:y + 3, x:x + 3]
        output_value = relu(np.sum(receptive_field * self.anim_kernel))
        self.anim_feature_map[y, x] = output_value
        self.draw_image(self.process_canvas, self.anim_input_image, "Convolving...", scan_box_pos=(x, y))
        self.draw_image(self.current_feature_map_canvas, self.anim_feature_map, "Building Feature Map...")

        x += 1
        if x >= self.anim_feature_map.shape[1]: x = 0; y += 1
        if y >= self.anim_feature_map.shape[0]:
            self.animation_timer.stop();
            self.finalize_convolution()
        else:
            self.anim_pos = (y, x)

    def finalize_convolution(self):
        tab_name = self.output_tabs.tabText(self.output_tabs.currentIndex())
        self.feature_maps[tab_name] = self.anim_feature_map
        self.draw_image(self.current_feature_map_canvas, self.anim_feature_map, f"Output: {tab_name}")
        self.draw_image(self.process_canvas, self.anim_input_image, "Convolution Complete")
        self.set_buttons_enabled(True)

    def draw_image(self, canvas, data, title, scan_box_pos=None):
        canvas.axes.clear();
        cmap = 'viridis'
        vmax = np.max(data) if np.max(data) > 0 else 1.0;
        vmin = 0
        if title.startswith("Input") or "Convolve" in title:
            cmap = 'gray_r';
            vmin = 0;
            vmax = 1.0
        canvas.axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        canvas.axes.set_title(title, fontsize=10)
        canvas.axes.set_xticks([]);
        canvas.axes.set_yticks([])
        if scan_box_pos:
            rect = Rectangle(xy=(scan_box_pos[0] - 0.5, scan_box_pos[1] - 0.5), width=3, height=3, edgecolor='red',
                             facecolor='red', alpha=0.3)
            canvas.axes.add_patch(rect)
        canvas.draw()

    def set_buttons_enabled(self, enabled):
        self.apply_to_input_button.setEnabled(enabled)
        self.apply_to_feature_map_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.clear_button.setEnabled(enabled)

    def clear_all_tabs(self):
        reply = QMessageBox.question(self, 'Confirm Clear',
                                     "Are you sure you want to clear all generated feature maps?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            # FIX: Simpler and more robust clearing logic
            while self.output_tabs.count() > 0:
                self.output_tabs.removeTab(0)
            self.feature_maps.clear()

    def save_all_maps(self):
        if self.output_tabs.count() == 0:
            QMessageBox.warning(self, "No Maps to Save", "Please generate at least one feature map before saving.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Images")

        if directory:
            saved_count = 0
            for i in range(self.output_tabs.count()):
                widget = self.output_tabs.widget(i)
                if isinstance(widget, MplCanvas):
                    tab_name = self.output_tabs.tabText(i)
                    sanitized_name = tab_name.replace(" + ", "_").replace(" ", "")
                    filePath = os.path.join(directory, f"{sanitized_name}.png")
                    widget.fig.savefig(filePath, dpi=300)
                    saved_count += 1

            QMessageBox.information(self, "Save Complete",
                                    f"Successfully saved {saved_count} feature map(s) to:\n{directory}")


# --- Main execution block ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
