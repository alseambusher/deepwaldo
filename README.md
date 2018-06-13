Deepwaldo
=========

Using [Mask-RCNN](https://github.com/matterport/Mask_RCNN) to solve Where's Waldo. Refer [this](https://lifepluslinux.blogspot.com) for detailed description of the problem and the solution.

Some results

![](data/val/5.jpg)
![](results/5.png)

## Data

Training and validation data sets used for this project are put in [data](data/) folder. For custom datasets, use [via-via](http://www.robots.ox.ac.uk/~vgg/software/via/via.html) tool to annotate.

## Running 

You can either set your configurations in the [waldo_config.py](waldo_config.py) file or use the default. Then use the following commands for training and predicting.

### Training

This will download the Mask-RCNN model trained on coco dataset to the MODEL_DIR folder and trains a model with the data in the DATA_DIR folder.

```bash

python train.py

```

### Detecting 

Shows a popup with the waldo detected in the image.


```bash
python predict.py [MODEL PATH] [PATH TO IMAGE]
# for example
python predict.py models/logs/waldo20180612T1628/mask_rcnn_waldo_0030.h5 data/val/5.jpg

```
 
