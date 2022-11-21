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
from ikomia.core.task import TaskParam
import copy
from ikomia.dnn import dnntrain
from train_unet.unet import UNet
from train_unet.train_model import train_net
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainUnetParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        # Place default value initialization here
        self.cfg["epochs"] = 50
        self.cfg["batch_size"] = 1
        self.cfg["learning_rate"] = 0.001
        self.cfg["val_percent"] = 10
        self.cfg["img_scale"] = 0.5
        self.cfg["num_channels"] = 3
        self.cfg["num_classes"] = 4
        self.cfg["outputFolder"] = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cfg["device"] = param_map["device"]
        self.cfg["inputSize"] = int(param_map["inputSize"])
        self.cfg["epochs"] = int(param_map["epochs"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["learning_rate"] = param_map["learning_rate"]
        self.cfg["val_percent"] = int(param_map["val_percent"])
        self.cfg["img_scale"] = param_map["img_scale"]
        self.cfg["num_channels"] = int(param_map["num_channels"])
        self.cfg["num_classes"] = int(param_map["num_classes"])
        self.cfg["outputFolder"] = param_map["outputFolder"]
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainUnet(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        self.stop_train = False

        # Create parameters class
        if param is None:
            self.setParam(TrainUnetParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        param = self.getParam()
        if param is not None:
            return param.cfg["epochs"]
        else:
            return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.problem = False
        self.stop_train = False
        # Get parameters :
        param = self.getParam()

        input = self.getInput(0)
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.problem = True

        # current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        # output dir
        if os.path.isdir(param.cfg["outputFolder"]):
            output_path = os.path.join(param.cfg["outputFolder"], str_datetime)
        else:
            # create output folder
            dir_path = os.path.dirname(__file__)
            output_path = os.path.join(dir_path, "output", str_datetime)
            os.makedirs(output_path)

        # tensorboard
        logdir = os.path.join(core.config.main_cfg["tensorboard"]["log_uri"], str_datetime)
        writer = SummaryWriter(logdir)

        # train the model
        model = UNet
        train_net(model, input.data, param.cfg["epochs"], param.cfg["batch_size"], param.cfg["learning_rate"],
              param.cfg["val_percent"], param.cfg["img_scale"], param.cfg["num_channels"], param.cfg["num_classes"],
                  param.cfg["outputFolder"],  self.get_stop, self.emitStepProgress, seed=10, writer = writer)

        # Call endTaskRun to finalize process
        self.endTaskRun()


    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train=True
# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainUnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_unet"
        self.info.shortDescription = "multi-class semantic segmentation using Unet"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "chaima bousnah"
        self.info.article = "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        self.info.year = 2015
        # Code source repository
        self.info.repository = "https://github.com/milesial/Pytorch-UNet"
        # Keywords used for search
        self.info.keywords = "semantic segmentation, unet, multi-class segmentation"

    def create(self, param=None):
        # Create process object
        return TrainUnet(self.info.name, param)
