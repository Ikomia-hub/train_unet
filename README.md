<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_unet/main/icon/unet.jpg" alt="Algorithm icon">
  <h1 align="center">train_unet</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_unet">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_unet">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_unet/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_unet.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train UNet model for semantic segmentation. 

![Unet car segmentation](https://camo.githubusercontent.com/0a5f6e3cb4ecc0b35b7af140ece691da92513e7dd53c5435155e4cab89d10cf7/68747470733a2f2f692e696d6775722e636f6d2f474438466342372e706e67)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_unet", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters


- **input_size** (int) - default '128': Size of the input image.
- **epochs** (int) - default '50': Number of complete passes through the training dataset.
- **batch_size** (int) - default '1': Number of samples processed before the model is updated.
- **learning_rate** (float) - default '0.001': Step size at which the model's parameters are updated during training.
- **val_percent** (int) – default '10': Divide the dataset into train and evaluation sets.
- **num_channels** (int) - default '3': Number of input chanel
- **output_folder** (str, *optional*): path to where the model will be saved. 


**Parameters** should be in **strings format**  when added to the dictionary.


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add dataset loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add training algorithm
train = wf.add_task(name="train_unet", auto_connect=True)
train.set_parameters({
    "batch_size": "1",
    "epochs": "50",
    "input_size": "128",
    "val_percent": "10",
    "learning_rate": "0.01",
    "num_channels": "3"
}) 

# Launch your training on your data
wf.run()
```


