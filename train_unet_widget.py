# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from train_unet.train_unet_process import TrainUnetParam

# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainUnetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = TrainUnetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()


        # image scale
        self.spin_scale = pyqtutils.append_double_spin(self.gridLayout, "img_scale", self.parameters.cfg["img_scale"])

        # num channels
        self.spin_channels = pyqtutils.append_spin(self.gridLayout, "num_channels", self.parameters.cfg["num_channels"])

        # Epochs
        self.spin_epochs = pyqtutils.append_spin(self.gridLayout, "epochs", self.parameters.cfg["epochs"])

        # batch size
        self.spin_batch = pyqtutils.append_spin(self.gridLayout, "batch_size", self.parameters.cfg["batch_size"])

        # learning rate
        self.spin_lr = pyqtutils.append_double_spin(self.gridLayout, "learning_rate", self.parameters.cfg["learning_rate"])

        # validation percentage
        self.spin_val = pyqtutils.append_spin(self.gridLayout, "val_percent", self.parameters.cfg["val_percent"])

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.gridLayout, label="Output folder",
                                                              path=self.parameters.cfg["outputFolder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)
        # PyQt -> Qt wrapping
        layout = qtconversion.PyQtToQt(self.gridLayout)
        self.setLayout(layout)


    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.cfg["img_scale"] = self.spin_scale.value()
        self.parameters.cfg["num_channels"] = self.spin_channels.value()
        self.parameters.cfg["epochs"] = self.spin_epochs.value()
        self.parameters.cfg["batch_size"] = self.spin_batch.value()
        self.parameters.cfg["learning_rate"] = self.spin_lr.value()
        self.parameters.cfg["val_percent"] = self.spin_val.value()
        self.parameters.cfg["outputFolder"] = self.browse_out_folder.path

        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainUnetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_unet"

    def create(self, param):
        # Create widget object
        return TrainUnetWidget(param, None)
