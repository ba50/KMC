import os
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog


from src.GUI import Ui_KMC_GUI
from src.GenerateXYZ import GenerateXYZ
from src.GenerateWorkers import GenerateWorkers
from src.TimeHeatMap import TimeHeatMap


def gen_sym(params_dict):
        if not (params_dict['path_to_data'] / 'heat_map').exists():
            (params_dict['path_to_data'] / 'heat_map').mkdir(parents=True, exist_ok=True)

        with (params_dict['path_to_data'] / 'input.kmc').open('w') as file_out:
            file_out.write("{}\t# Cell type\n".format(params_dict['cell_type'].lower()))
            file_out.write("{}\t# X number of cells\n".format(params_dict['size'][0]))
            file_out.write("{}\t# Y number of cells\n".format(params_dict['size'][1]))
            file_out.write("{}\t# Z number of cells\n".format(params_dict['size'][2]))
            file_out.write("{}\t# Thermalization time\n".format(params_dict['thermalization_time']))
            file_out.write("{}\t# Time to end simulation\n".format(params_dict['time_end']))
            file_out.write("{}\t# Left contact switch\n".format(params_dict['contact_switch'][0]))
            file_out.write("{}\t# Right contact switch\n".format(params_dict['contact_switch'][1]))
            file_out.write("{}\t# Left contact\n".format(params_dict['contact'][0]))
            file_out.write("{}\t# Right contact\n".format(params_dict['contact'][1]))
            file_out.write("{}\t# Amplitude of sine function\n".format(params_dict['energy_params'][0]))
            file_out.write("{}\t# Frequency base of sine function\n".format(params_dict['energy_params'][1]))
            file_out.write("{}\t# Frequency power of sine function\n".format(params_dict['energy_params'][2]))
            file_out.write("{}\t# Period of sine function\n".format(params_dict['energy_params'][3]))
            file_out.write("{}\t# Delta energy base\n".format(params_dict['energy_params'][4]))

        if params_dict['cell_type'] == 'Random':
            GenerateXYZ(params_dict['size'], params_dict['path_to_data']).generate_random()
        if params_dict['cell_type'] == 'Sphere':
            GenerateXYZ(self.params_dict['size'], params_dict['path_to_data']).generate_sphere(5)
        if params_dict['cell_type'] == 'Plane':
            GenerateXYZ(params_dict['size'], params_dict['path_to_data']).generate_plane(1)



class Launcher(Ui_KMC_GUI):
    cell_type = None
    root_directory = None
    path_to_data = None
    size = None
    thermalization_time = None
    time_end = None
    contact_switch = None
    contact = None
    energy_param = None
    models = []
    root = None
    time_heatmap = None

    def __init__(self):
        self.KMC_GUI = QtWidgets.QMainWindow()
        self.setupUi(self.KMC_GUI)

        self.root_directory = os.getcwd()
        self.label_path_to_data.setText(self.root_directory)

        self.pushButton_path_to_data.clicked.connect(self.handleButton_path_to_data)
        self.pushButton_generate.clicked.connect(self.handleButton_generate)

        self.pushButton_add_model.clicked.connect(self.handleButton_add_model)
        self.pushButton_clear_all.clicked.connect(self.handleButton_clear_all)
        self.pushButton_launch.clicked.connect(self.handleButton_launch)

        self.pushButton_add_root.clicked.connect(self.handleButton_add_root)
        self.pushButton_plot.clicked.connect(self.handleButton_plot)
        self.pushButton_heatmap.clicked.connect(self.handleButton_heatmap)

    def handleButton_path_to_data(self):
        self.root_directory = QFileDialog.getExistingDirectory(self.centralWidget)
        self.label_path_to_data.setText(self.root_directory)

    def handleButton_generate(self):
        self.cell_type = self.comboBox_type.currentText()

        self.size = self.spinBox_x_size.value(), self.spinBox_y_size.value(), self.spinBox_z_size.value()

        self.time_end = self.spinBox_time_end.value()
        self.thermalization_time = self.spinBox_thermalization_time.value()

        self.contact_switch = self.checkBox_left_contact.checkState(), self.checkBox_right_contact.checkState()
        self.contact = self.spinBox_left_contact.value(), self.spinBox_right_contact.value()

        self.enegry_param = (self.doubleSpinBox_A.value(),
                             self.doubleSpinBox_frequency_base.value(),
                             self.spinBox_frequency_power.value(),
                             self.spinBox_period.value(),
                             self.doubleSpinBox_delta_energi_base.value())

        self.path_to_data = Path(self.root_directory,
                                 str(self.size[0])+'_' +
                                 str(self.size[1])+'_' +
                                 str(self.size[2])+'_' +
                                 self.cell_type.lower())
        if not (self.path_to_data / 'heat_map').exists():
            (self.path_to_data / 'heat_map').mkdir(parents=True, exist_ok=True)

        with (self.path_to_data / 'input.kmc').open('w') as file_out:
            file_out.write("{}\t# Cell type\n".format(self.cell_type.lower()))
            file_out.write("{}\t# X number of cells\n".format(self.size[0]))
            file_out.write("{}\t# Y number of cells\n".format(self.size[1]))
            file_out.write("{}\t# Z number of cells\n".format(self.size[2]))
            file_out.write("{}\t# Thermalization time\n".format(self.thermalization_time))
            file_out.write("{}\t# Time to end simulation\n".format(self.time_end))
            file_out.write("{}\t# Left contact switch\n".format(self.contact_switch[0]))
            file_out.write("{}\t# Right contact switch\n".format(self.contact_switch[1]))
            file_out.write("{}\t# Left contact\n".format(self.contact[0]))
            file_out.write("{}\t# Right contact\n".format(self.contact[1]))
            file_out.write("{}\t# Amplitude of sine function\n".format(self.enegry_param[0]))
            file_out.write("{}\t# Frequency base of sine function\n".format(self.enegry_param[1]))
            file_out.write("{}\t# Frequency power of sine function\n".format(self.enegry_param[2]))
            file_out.write("{}\t# Period of sine function\n".format(self.enegry_param[3]))
            file_out.write("{}\t# Delta energy base\n".format(self.enegry_param[4]))

        if self.cell_type == 'Random':
            GenerateXYZ(self.size, self.path_to_data).generate_random()
        if self.cell_type == 'Sphere':
            GenerateXYZ(self.size, self.path_to_data).generate_sphere(5)
        if self.cell_type == 'Plane':
            GenerateXYZ(self.size, self.path_to_data).generate_plane(1)

    def handleButton_clear_all(self):
        self.models = []
        self.label_models.setText('\n'.join(self.models))

    def handleButton_add_model(self):
        self.models.append(str(QFileDialog.getExistingDirectory(self.centralWidget)))
        self.label_models.setText('\n'.join(self.models))

    def handleButton_launch(self):
        commands = []
        for model in self.models:
            commands.append('./build/KMC.exe ' + model)

        threads = self.spinBox_threads.value()

        print(commands)
        workers_pool = GenerateWorkers(commands, threads)
        workers_pool.run()

    def handleButton_add_root(self):
        self.root = Path(QFileDialog.getExistingDirectory(self.centralWidget))
        self.label_roots.setText(self.root)
        self.time_heatmap = TimeHeatMap(self.root)

    def handleButton_heatmap(self):
        self.time_heatmap.save_heatmap()

    def handleButton_plot(self):
        self.time_heatmap.plot_layer_in_time(self.spinBox_layer.value())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    launcher = Launcher()
    launcher.KMC_GUI.show()
    sys.exit(app.exec_())
