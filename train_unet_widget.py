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
        # PyQt -> Qt wrapping
        layout = qtconversion.PyQtToQt(self.gridLayout)

        img_scaleLabel = QLabel("Scale (resize images)")
        self.img_scaleSpinBox = QDoubleSpinBox()
        self.img_scaleSpinBox.setRange(0.1, 1)
        self.img_scaleSpinBox.setDecimals(4)
        self.img_scaleSpinBox.setSingleStep(0.0001)
        self.img_scaleSpinBox.setValue(self.parameters.cfg["img_scale"])

        num_classesLabel = QLabel("Classes number:")
        self.num_classesSpinBox = QSpinBox()
        self.num_classesSpinBox.setRange(1, 2147483647)
        self.num_classesSpinBox.setSingleStep(1)
        self.num_classesSpinBox.setValue(self.parameters.cfg["num_classes"])

        num_channelsLabel = QLabel("Channels number:")
        self.num_channelsSpinBox = QSpinBox()
        self.num_channelsSpinBox.setRange(1, 4)
        self.num_channelsSpinBox.setSingleStep(1)
        self.num_channelsSpinBox.setValue(self.parameters.cfg["num_channels"])

        epochsLabel = QLabel("Number of epochs:")
        self.epochsSpinBox = QSpinBox()
        self.epochsSpinBox.setRange(1, 2147483647)
        self.epochsSpinBox.setSingleStep(1)
        self.epochsSpinBox.setValue(self.parameters.cfg["epochs"])

        batch_sizeLabel = QLabel("Batch size:")
        self.batch_sizeSpinBox = QSpinBox()
        self.batch_sizeSpinBox.setRange(1, 2147483647)
        self.batch_sizeSpinBox.setSingleStep(1)
        self.batch_sizeSpinBox.setValue(self.parameters.cfg["batch_size"])

        learning_rateLabel = QLabel("learning rate:")
        self.learning_rateSpinBox = QDoubleSpinBox()
        self.learning_rateSpinBox.setRange(0, 10)
        self.learning_rateSpinBox.setDecimals(4)
        self.learning_rateSpinBox.setSingleStep(0.0001)
        self.learning_rateSpinBox.setValue(self.parameters.cfg["learning_rate"])

        val_percentLabel = QLabel("validation percentage:")
        self.val_percentSpinBox = QSpinBox()
        self.val_percentSpinBox.setRange(0, 100)
        self.val_percentSpinBox.setSingleStep(1)
        self.val_percentSpinBox.setValue(self.parameters.cfg["val_percent"])


        # Set widget layout
        self.gridLayout.addWidget(img_scaleLabel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.img_scaleSpinBox, 0, 1, 1, 2)
        self.gridLayout.addWidget(num_classesLabel, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.num_classesSpinBox, 1, 1, 1, 2)
        self.gridLayout.addWidget(num_channelsLabel, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.num_channelsSpinBox, 2, 1, 1, 2)

        self.gridLayout.addWidget(epochsLabel, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.epochsSpinBox, 3, 1, 1, 2)
        self.gridLayout.addWidget(batch_sizeLabel, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.batch_sizeSpinBox, 4, 1, 1, 2)
        self.gridLayout.addWidget(learning_rateLabel, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.learning_rateSpinBox, 5, 1, 1, 2)
        self.gridLayout.addWidget(val_percentLabel, 6, 0, 1, 1)
        self.gridLayout.addWidget(self.val_percentSpinBox, 6, 1, 1, 2)

        # Output folder
        self.browse_out_folder = pyqtutils.append_browse_file(self.gridLayout, label="Output folder",
                                                              path=self.parameters.cfg["outputFolder"],
                                                              tooltip="Select folder",
                                                              mode=QFileDialog.Directory)

        self.setLayout(layout)


    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.cfg["img_scale"] = self.img_scaleSpinBox.value()
        self.parameters.cfg["num_classes"] = self.num_classesSpinBox.value()
        self.parameters.cfg["num_channels"] = self.num_channelsSpinBox.value()
        self.parameters.cfg["epochs"] = self.epochsSpinBox.value()
        self.parameters.cfg["batch_size"] = self.batch_sizeSpinBox.value()
        self.parameters.cfg["learning_rate"] = self.learning_rateSpinBox.value()
        self.parameters.cfg["val_percent"] = self.val_percentSpinBox.value()
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
