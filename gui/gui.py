import sys
sys.path.append('../EEG-ECoG') # adding path for packages
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QMessageBox, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QSpacerItem, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mne
from mne.datasets import sample
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
class LinearRegressionModel(nn.Module):
    def init(self, input_size, output_size):
        super(LinearRegressionModel, self).init()
        self.linear = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(64, output_size)

    def forward(self, x):
        x1 = self.linear(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3

class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG to EcoG 0.0.1")
        self.setGeometry(50, 50, 800, 700)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout(self.central_widget)

        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_change)
        layout.addWidget(self.tab_widget)

        # Create the Temperature-Entropy (TS) tab
        self.graph_tab = QWidget()
        self.layout_graphtab = QHBoxLayout(self.graph_tab)
        self.more_tab = QWidget()
        self.layout_moretab = QVBoxLayout(self.more_tab)

        self.bar_graph_figure = plt.figure()
        self.bar_graph_ax = self.bar_graph_figure.add_subplot(111)
        self.bar_graph_canvas = FigureCanvas(self.bar_graph_figure)
        self.bar_graph_title = ''

        self.left_graph_figure = plt.figure()
        self.left_graph_ax = self.left_graph_figure.add_subplot(111)
        self.left_graph_canvas = FigureCanvas(self.left_graph_figure)
        self.left_graph_title = ''

        self.right_graph_figure = plt.figure()
        self.right_graph_ax = self.right_graph_figure.add_subplot(111)
        self.right_graph_canvas = FigureCanvas(self.right_graph_figure)
        self.right_graph_title = ''

        self.bar_graph_canvas.setFixedSize(500, 400)
        self.left_graph_canvas.setFixedSize(500, 400)
        self.right_graph_canvas.setFixedSize(500, 400)

        self.layout_moretab.addWidget(self.bar_graph_canvas)
        self.layout_graphtab.addWidget(self.left_graph_canvas)
        self.layout_graphtab.addWidget(self.right_graph_canvas)

        self.graph_tab.setLayout(self.layout_graphtab)
        self.more_tab.setLayout(self.layout_moretab)
        self.tab_widget.addTab(self.graph_tab, "Graph")
        self.tab_widget.addTab(self.more_tab, "More")

        self.transformed_data = None

        self.init_inputs_section()
        # self.display_model()

        self.model = torch.load('models/linear_model_final.pth')

    def on_tab_change(self, index):
        tab_text = self.tab_widget.tabText(index)
        if tab_text == "More":
            self.plot_bargraph()

    def plot_bargraph(self):
        channels = [i for i in range(1, 19)]
        prediction_rates = [0.56037093, 0.57641234, 0.55711894, 0.5016344,  0.50797424, 0.55253317,
 0.55874642, 0.580673,   0.58669681, 0.58267718, 0.55131443, 0.51515929,
 0.53597434, 0.56285037, 0.49295108, 0.58599707, 0.55752726, 0.56338094]
        self.bar_graph_ax.clear()
        self.bar_graph_ax.bar(channels, prediction_rates, width=0.65)
        self.bar_graph_ax.set_xlabel('Channels')
        self.bar_graph_ax.set_ylabel('Prediction Rate')
        self.bar_graph_ax.set_title('Prediction Rates by Channel')
        self.bar_graph_canvas.draw()

    def display_model(self):
        mne.set_log_level(verbose='ERROR')
        data_path = sample.data_path()
        subjects_dir = data_path / "subjects"
        sample_dir = data_path / "MEG" / "sample"

        brain_kwargs = dict(alpha=0.1, background="white", cortex="low_contrast")
        brain = mne.viz.Brain("sample", subjects_dir=subjects_dir, **brain_kwargs)

        stc = mne.read_source_estimate(sample_dir / "sample_audvis-meg")
        stc.crop(0.09, 0.1)

        kwargs = dict(
            fmin=stc.data.min(),
            fmax=stc.data.max(),
            alpha=0.25,
            smoothing_steps="nearest",
            time=stc.times,
        )
        brain.add_data(stc.lh_data, hemi="lh", vertices=stc.lh_vertno, **kwargs)
        brain.add_data(stc.rh_data, hemi="rh", vertices=stc.rh_vertno, **kwargs)

        brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
        brain.add_label("BA44", hemi="lh", color="green", borders=True)
        brain.show_view(azimuth=190, elevation=70, distance=350, focalpoint=(0, 0, 20))
        brain.add_head(alpha=0.5)

        evoked = mne.read_evokeds(sample_dir / "sample_audvis-ave.fif", verbose=False)[0]
        trans = mne.read_trans(sample_dir / "sample_audvis_raw-trans.fif")
        brain.add_sensors(evoked.info, trans)
        brain.show_view(distance=500)

        dip = mne.read_dipole(sample_dir / "sample_audvis_set1.dip")
        cmap = plt.get_cmap("YlOrRd")
        colors = [cmap(gof / dip.gof.max()) for gof in dip.gof]
        brain.add_dipole(dip, trans, colors=colors, scales=list(dip.amplitude * 1e8))
        brain.show_view(azimuth=-20, elevation=60, distance=300)

    def init_inputs_section(self):
        inputs_widget = QWidget()
        inputs_layout = QVBoxLayout(inputs_widget)

        hbox_csv_button = QHBoxLayout()
        spacer_left = QSpacerItem(250, 0)
        spacer_right = QSpacerItem(250, 0)
        self.csv_button = QPushButton("Select CSV File")
        self.csv_button.setStyleSheet("border-radius: 5px; border: 1px solid gray")
        self.csv_button.setFixedWidth(150)
        self.csv_button.clicked.connect(self.get_csv_file)
        # hbox_csv_button.addItem(spacer_left)
        hbox_csv_button.addWidget(self.csv_button)
        # hbox_csv_button.addItem(spacer_right)
        inputs_layout.addLayout(hbox_csv_button)

        # Placeholder for save button, initially hidden
        spacerhwbow = QHBoxLayout()
        spacer_left = QSpacerItem(250, 0)
        spacer_right = QSpacerItem(250, 0)
        self.save_button = QPushButton("Save Data")
        self.save_button.setStyleSheet("border-radius: 5px; border: 1px solid gray")
        self.save_button.setFixedWidth(150)
        # spacerhwbow.addItem(spacer_left)
        spacerhwbow.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_data)
        self.save_button.setVisible(False)
        # spacerhwbow.addItem(spacer_right)
        inputs_layout.addLayout(spacerhwbow)

        self.central_widget.layout().addWidget(inputs_widget)

    def get_csv_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv);;All Files (*)")
        if file_name:
            self.read_and_plot_csv(file_name)

    def read_and_plot_csv(self, file_path):
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                raise ValueError("The CSV file is empty.")
            # print(df)
            self.convert(df)
            self.plot_csv_original_data(df)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read or plot CSV file: {e}")

    def plot_csv_original_data(self, df):
        self.left_graph_ax.clear()
        x = df.index
        print(df)
        for column in df.columns:
            y = df[column]
            self.left_graph_ax.plot(x, y, marker='.', linestyle='-')

        self.left_graph_ax.set_xlabel('Row Index')
        self.left_graph_ax.set_ylabel('Values')
        self.left_graph_ax.set_title('ECoG Data')
        self.left_graph_ax.legend()
        self.left_graph_canvas.draw()

    def convert(self, file):
        self.model.eval()
        data = torch.tensor(np.array(file)[:,1:].astype(float), dtype=torch.float32)
        column = np.array(file)[:,0]
        y_pred = self.model(data.T)
        self.plot_csv_transformed_data(y_pred, column)

    def plot_csv_transformed_data(self, df, columns_name):
        self.right_graph_ax.clear()
        for column in range(df.shape[1]):
            y = df[:,column]
            self.right_graph_ax.plot(np.arange(df.shape[0]), y.detach().numpy(), marker='.', linestyle='-')

        self.transformed_data = df
        self.right_graph_ax.set_xlabel('Row Index')
        self.right_graph_ax.set_ylabel('Values')
        self.right_graph_ax.set_title('EEG Data')
        self.right_graph_ax.legend()
        self.right_graph_canvas.draw()
        self.save_button.setVisible(True)

    def save_data(self):
        # Prompt user for file save location and name
        file_path, _ = QFileDialog.getSaveFileName(None,
                    "Save File", "", "CSV Files (*.csv);;Excel Files (*.xlsx);;Text Files (*.txt);;JSON Files (*.json)")
        if self.transformed_data is not None:
            print(self.transformed_data)
            transformed_array = self.transformed_data.detach().numpy()
            num_rows, num_cols = transformed_array.shape
            df = pd.DataFrame(transformed_array)
            # Save the file as CSV
            df.to_csv(file_path, index=True)
            print(f"Saved {file_path} as CSV")

    def on_button_click(self, button):
        if self.current_button_mat is not None:
            self.make_button_normal(self.current_button_mat)
        self.make_button_gray(button, "#A9A9A9")

    def make_button_gray(self, button, color):
        button.setStyleSheet(f"background-color: {color}; border-radius: 5px; border: 1px solid gray")
        button.setMinimumHeight(30)
        self.current_button_mat = button

    def make_button_normal(self, button):
        button.setStyleSheet("border-radius: 5px; border: 1px solid gray")
        self.current_button_mat = button

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphWindow()
    window.show()
    sys.exit(app.exec())