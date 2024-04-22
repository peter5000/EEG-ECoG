import sys
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QMessageBox, QVBoxLayout, QHBoxLayout, QGridLayout, QTextEdit, QLabel,
    QPushButton, QComboBox, QLineEdit, QCheckBox, QFileDialog, QSizePolicy, QHeaderView
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class GraphWindow(QMainWindow):
    def __init__(self):
        super().__init__()
 
        self.setWindowTitle("EEG to EcoG 0.0.1")
        self.setGeometry(200, 150, 1000, 700)
        # self.setStyleSheet("border-radius: 10px; border: 1px solid gray")
 
        self.central_widget = QWidget(self)
        # background_image = QPixmap("art.jpg")
        # self.central_widget.setStyleSheet(f"background-image: url({'art'}); background-repeat: no-repeat; background-attachment: fixed;")
        self.setCentralWidget(self.central_widget)
        layout = QHBoxLayout(self.central_widget)

        # Create the tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
 
        # Create the Temperature-Entropy (TS) tab
        self.graph_tab = QWidget()
        self.graph_figure = plt.figure()
        self.graph_ax = self.graph_figure.add_subplot(111, projection = '3d')
        self.graph_canvas = FigureCanvas(self.graph_figure)
        self.graph_title = ''
        self.conversion = ''
        self.prev_conversion = ''

        self.current_button_mat = None

        # self.graph_canvas.setMinimumSize(500, 500)
        self.layout_graphtab = QVBoxLayout(self.graph_tab)
        self.layout_graphtab.addWidget(self.graph_canvas)
 
        self.x_input_boxes = [QLineEdit() for _ in range(3)]
        self.y_input_boxes = [QLineEdit() for _ in range(3)]
        
        for x_input_box in self.x_input_boxes:
            x_input_box.setStyleSheet("border-radius: 5px; border: 1px solid gray")

        for y_input_box in self.y_input_boxes:
            y_input_box.setStyleSheet("border-radius: 5px; border: 1px solid gray")
 
        # Initialize previous values
        self.prev_nX = 100
        self.prev_nY = 100
        self.prev_Xmin = 1
        self.prev_Xmax = 100
        self.prev_Ymin = 1
        self.prev_Ymax = 100
 
        # Tab Group
        self.init_graph_tab()
        self.init_more_tab()
        self.init_inputs_section()

    ## MAIN GRAPH
    def init_graph_tab(self):
        self.tab_widget.addTab(self.graph_tab, "Graph")

    ## MORE
    def init_more_tab(self):
        # Create a "More" tab
        more_tab = QWidget()
        more_layout = QVBoxLayout(more_tab)
        # Create a "Help" button
        help_button = QPushButton("Help")
        help_button.clicked.connect(self.open_github_page)
        help_button.setFixedWidth(100)
        more_layout.addWidget(help_button)
        # Create a "Help" button
        ref_button = QPushButton("References")
        ref_button.clicked.connect(self.open_references_page)
        more_layout.addWidget(ref_button)
        ref_button.setFixedWidth(100)
        more_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.tab_widget.addTab(more_tab, "More")
    
    def open_github_page(self):
        # Define the URL to your GitHub repository
        github_url = "https://github.com/Bjournaux/SeaFreeze/tree/master/SeaFreezeGUI"
 
        # Open the URL in the default web browser
        QDesktopServices.openUrl(QUrl(github_url))
    
    def open_references_page(self):
        url = 'https://github.com/Bjournaux/SeaFreeze'
        # webbrowser.open(url, new=2)  # 'new=2' opens the URL in a new browser window or tab
        QDesktopServices.openUrl(QUrl(url))
    
    def update_graph(self):
        if not self.conversion:
            return
        
        self.title = self.conversion

        # Retrieve current input values
        self.current_nX = int(self.x_input_boxes[2].text()) if self.x_input_boxes[2].text() else 100
        self.current_nY = int(self.y_input_boxes[2].text()) if self.y_input_boxes[2].text() else 100
        self.current_Xmin = np.double(self.x_input_boxes[0].text()) if self.x_input_boxes[0].text() else 0
        self.current_Xmax = np.double(self.x_input_boxes[1].text()) if self.x_input_boxes[1].text() else 400
        self.current_Ymin = np.double(self.y_input_boxes[0].text()) if self.y_input_boxes[0].text() else 240
        self.current_Ymax = np.double(self.y_input_boxes[1].text()) if self.y_input_boxes[1].text() else 270
        
        # If values haven't been changed
        if (self.current_nX, self.current_nY, self.current_Xmin, self.current_Xmax,
            self.current_Ymin, self.current_Ymax) == (self.prev_nX, self.prev_nY,
            self.prev_Xmin, self.prev_Xmax, self.prev_Ymin, self.prev_Ymax):
            # And if material hasn't been changed
            if self.conversion == self.prev_conversion:
                return
        
        # Update previous values with current values
        self.prev_nX = self.current_nX if self.current_nX else None
        self.prev_nY = self.current_nY if self.current_nY else None
        self.prev_Xmin = self.current_Xmin if self.current_Xmin else None
        self.prev_Xmax = self.current_Xmax if self.current_Xmax else None
        self.prev_Ymin = self.current_Ymin if self.current_Ymin else None
        self.prev_Ymax = self.current_Ymax if self.current_Ymax else None
        
        invalid_currents = (self.current_Xmin == None or self.current_Xmax == None) or (
                            self.current_Ymin == None or self.current_Ymax == None or (
                            self.current_nX == None) or (self.current_nY == None))
        
        match self.conversion:
            case 'EEG -> EcoG':
                valid_nums = (0 <= self.current_Pmin <= self.current_Pmax <= 400
                                and 1 <= self.current_Tmin <= self.current_Tmax <= 301)
                if not valid_nums or invalid_currents:
                    return
            case 'EcoG -> EEG':
                valid_nums = (0 <= self.current_Pmin <= self.current_Pmax <= 900
                                and 0 <= self.current_Tmin <= self.current_Tmax <= 270)
                if not valid_nums or invalid_currents:
                    return
            
        # Define the PT conditions
        if (self.current_Pmin and self.current_Pmax and self.current_nP) and (self.current_Tmin and self.current_Tmax and self.current_nT):
            X = np.arange(self.current_Pmin, self.current_Pmax, (self.current_Pmax-self.current_Pmin)/self.current_nP)
            Y = np.arange(self.current_Tmin, self.current_Tmax, (self.current_Tmax-self.current_Tmin)/self.current_nT)
            Z = np.array([X, Y], dtype='object')
        
        if self.current_nP == 1 or self.current_nT == 1:
            # Replace the existing axis (if it's 3D)
            if isinstance(self.graph_ax, Axes3D):
                self.graph_ax.remove()
                self.graph_ax = self.graph_figure.add_subplot(111)
                self.graph_ax.clear()
            elif isinstance(self.graph_ax, Axes):
                self.graph_ax.clear()
            self.graph_figure.set_size_inches(7, 7)
            self.graph_figure.subplots_adjust(left=0.2, right=1, bottom=0.2, top=0.9)
            if self.current_nP == 1:
                self.graph_ax.plot(Y, Z)  # Plot temperature vs. data
                self.graph_ax.set_xlabel('Y', fontsize=10, labelpad=10)
                self.graph_ax.set_ylabel(self.conversion, fontsize=10, labelpad=10)
            elif self.current_nT == 1:
                self.graph_ax.plot(X, Z)  # Plot pressure vs. data
                self.graph_ax.set_xlabel('X', fontsize=10, labelpad=10)
                self.graph_ax.set_ylabel(self.mat, fontsize=10, labelpad=10)
            else:
                return
            self.graph_ax.set_title(self.title)
            self.graph_canvas.draw()
        else:
            # Replace the existing axis (if it's 2D)
            if isinstance(self.graph_ax, Axes):
                self.graph_ax.remove()
                self.graph_ax = self.graph_figure.add_subplot(111, projection='3d')
                self.graph_ax.clear()
            elif isinstance(self.graph_ax, Axes3D):
                self.graph_ax.clear()
            # Create a grid of P and T values for the surface plot
            X_grid, Y_grid = np.meshgrid(X, Y)
            self.graph_figure.set_size_inches(7, 7)
            self.graph_figure.subplots_adjust(left=0.1, right=1, bottom=0.1, top=0.9)
            self.graph_ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis')
            self.graph_ax.set_xlabel('X', fontsize=10, labelpad=10)
            self.graph_ax.set_ylabel('Y', fontsize=10, labelpad=10)
            self.graph_ax.set_zlabel('Z', fontsize=10, labelpad=10)
            self.graph_ax.set_title(self.graph_title, fontsize = 12)
            self.graph_canvas.draw()
        self.points = list(zip(X.ravel(), Y.ravel(), Z.ravel()))

    def update_conversion(self, index):
        conv = self.dropdown.currentText()
        self.conversion = conv
        self.graph_title = self.dropdown.itemText(index)
    
    ## INPUTS SECTION
    def init_inputs_section(self):
        inputs_widget = QWidget()
        inputs_layout = QVBoxLayout(inputs_widget)       
        
        ## SELECT MATERIAL DROPDOWN
        dropdown_layout = QHBoxLayout()
        dropdown_layout.addSpacing(-30)
        dropdown_label = QLabel("Select Conversion:")
        dropdown_label.setFixedWidth(130)
        select_material_font = dropdown_label.font()
        select_material_font.setPointSize(12)
        select_material_font.setFamily("Arial")
        dropdown_label.setFont(select_material_font)
        dropdown_layout.addWidget(dropdown_label)
        
        # Create and add the dropdown to the layout
        self.dropdown = QComboBox()
        self.dropdown.addItems(['Select', 'EEG -> EcoG', 'EcoG -> EEG'])
        self.dropdown.setFixedWidth(110)
        self.dropdown.setFixedHeight(25)
        self.dropdown.setStyleSheet("QComboBox { border-radius: 5px; border: 1px solid gray;" + 
                                    "background-color: transparent}")
        dropdown_layout.addWidget(self.dropdown)
        
        ### BUTTONS SECTION
        self.button_layout = QGridLayout() # data type buttons
        button_labels = ["button 1", "button 2", "button 3", "button 4", "button 5",
                         "button 6", "button 7", "button 8"]
        
        buttons_per_column_ = 4

        # ADD DATA TYPE BUTTONS TO BUTTON LAYOUT
        for i, label in enumerate(button_labels):
            row = i % buttons_per_column_
            col = i // buttons_per_column_
 
            button = QPushButton(label)
            button.setFixedWidth(150)
            button.setMinimumHeight(30)
            button.setStyleSheet("border-radius: 5px; border: 1px solid gray")
            self.button_layout.addWidget(button, row, col)
            # Connect the button click event to update_graph function with the corresponding label
            button.clicked.connect(lambda _, button=button: self.on_button_click(button))

        # Connect the dropdown's signal to the update_mat slot
        self.dropdown.currentIndexChanged.connect(self.update_conversion)     

        input_values_layout = QHBoxLayout()
        input_values_layout.addSpacing(150)
        in_vals_label = QLabel("Input Values:")
        in_vals_label.setFixedWidth(230)
        in_vals_label.setFixedHeight(25)
        input_values_layout.addWidget(in_vals_label)
        
        x_input_layout = QVBoxLayout()
        x_input_layout.addSpacing(-150)
        x_input_labels = ["Xmin:", "Xmax:", "nX:"]
        x_unit_labels = ["", "", ""]

        # Add pressure input boxes to pressure layout
        for x_label, x_input_box, x_unit_label in zip(x_input_labels, self.x_input_boxes, x_unit_labels):
            hbox = QHBoxLayout()
            hbox.setSpacing(50)
            hbox.setContentsMargins(10, 0, 10, 0)
            x_label_widget = QLabel(x_label)
            x_label_widget.setFixedSize(50, 50)
            x_label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hbox.addWidget(x_label_widget)
            x_input_box.setMaximumWidth(65)
            hbox.addSpacing(-40)
            hbox.addWidget(x_input_box)
            hbox.addSpacing(-45)
            x_unit_label_widget = QLabel(x_unit_label)
            x_unit_label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            x_unit_label_widget.setFixedSize(40, 40)
            hbox.addWidget(x_unit_label_widget)
            x_input_layout.addLayout(hbox)
 
        y_input_layout = QVBoxLayout()
        y_input_layout.addSpacing(-150)
        y_input_labels = ["Ymin:", "Ymax:", "nY:"]
        y_unit_labels = ["", "", ""]

        # Add pressure input boxes to pressure layout
        for y_label, y_input_box, y_unit_label in zip(y_input_labels, self.y_input_boxes, y_unit_labels):
            hbox = QHBoxLayout()
            hbox.setSpacing(50)
            hbox.setContentsMargins(10, 0, 10, 0)
            y_label_widget = QLabel(y_label)
            y_label_widget.setFixedSize(50, 50)
            y_label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hbox.addWidget(y_label_widget)
            y_input_box.setMaximumWidth(65)
            hbox.addSpacing(-40)
            hbox.addWidget(y_input_box)
            hbox.addSpacing(-45)
            y_unit_label_widget = QLabel(y_unit_label)
            y_unit_label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            y_unit_label_widget.setFixedSize(40, 40)
            hbox.addWidget(y_unit_label_widget)
            y_input_layout.addLayout(hbox)
            
        combined_input_layout = QHBoxLayout()
        combined_input_layout.addLayout(x_input_layout)
        combined_input_layout.addSpacing(-50)
        combined_input_layout.addLayout(y_input_layout)
        
        inputs_layout.addSpacing(-80)
        inputs_layout.addLayout(dropdown_layout)
        inputs_layout.addSpacing(-60)
        inputs_layout.addLayout(self.button_layout)
        inputs_layout.addSpacing(-50)
        inputs_layout.addLayout(input_values_layout)
        inputs_layout.addSpacing(50)
        inputs_layout.addLayout(combined_input_layout)
 
        # Update graph button
        hbox_update_button = QHBoxLayout()
        self.update_button = QPushButton("Update Graph")
        self.update_button.setStyleSheet("border-radius: 5px; border: 1px solid gray")
        self.update_button.setFixedWidth(100)
        self.update_button.clicked.connect(self.update_graph)
        hbox_update_button.addWidget(self.update_button)
        inputs_layout.addSpacing(-50)
        inputs_layout.addLayout(hbox_update_button)
        
        # Add the inputs widget to the main layout (central_widget)
        self.central_widget.layout().addWidget(inputs_widget)
        return inputs_widget
    
    def on_button_click(self, button):
        # Check if there's a previously clicked button and make it normal
        if self.current_button_mat is not None:
            self.make_button_normal(self.current_button_mat)

        # Make the clicked button gray
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
