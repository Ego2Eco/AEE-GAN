# AEE-GAN
### This is the code for the paper<br />
### Trajectory Prediction in Heterogeneous Environment via Attended Ecology Embedding<br />
Presented at [ACMMM 2020](https://2020.acmmm.org/)

## Introduction
To predict convincing trajectories in the heterogeneous environment remains challenging due to the complicated social interactions and physical constraints in a scene. We tackle these problems by introducing two enforced attention modules to socially and visually attend the important information from ecology.<br />
Below is how our model performs in the heterogeneous environment. The ground truth is colored in blue while the predicted trajectory of our model is colored in green.

![image](https://github.com/Ego2Eco/AEE-GAN/blob/master/figs/15.gif)

## System Architecture
This is an overview of our proposed AEE-GAN architecture which consists of three key module: 
- Feature Encoder 
- Enforced Attention 
- LSTM-based Info-GAN <br />

![image](https://github.com/Ego2Eco/AEE-GAN/blob/master/figs/AEE-GAN.png)

## Installation
This work is developed and tested on Ubuntu 18.04 with Python 3.7.3 and Pytorch 1.2.0. We have also used CUDA and cuDNN to speed up our model.

1. Clone this repository. 

        git clone 
2. Setup virtual environment:
   
    You can create the virtual environment from `environment.yaml` file. 

        conda env create -f environment.yaml

## Train
You need to edit the **input/output part** in the `train.py` file to select which dataset you want to use for training. After editing then you can start training.

    python3 train.py
Details about datasets can be found below. 
## Datasets
We trained this model on both heterogeneous environment datasets and homogeneous environment datasets. 
1. [Waymo Open Dataset](https://waymo.com/open/): 

    Waymo dataset is recently released and consists of front-view images of various real-world road-agents (i.e., vehicles, pedestrians and bikes).
    The coordinates of Waymo dataset are provided in the world coordinate.

    Due to Waymo's license agreement, we cannot upload modified dataset or model trained on it. Please download the Waymo dataset by yourself and follow steps below to transform the dataset into useable format for this model.

    1. aaa
    2. bbb

2. [Stanford Drone Dataset (SDD)](https://cvgl.stanford.edu/projects/uav_data/):
   
    Stanford Drone Dataset is a standard benchmark for trajectory prediction containing the categories of the road-agents. Different from Waymo dataset, the images are provided from the top-view angle and the coordinates are provided in pixel.

    We have uploaded this dataset which we make some modification to it. If you want to use it for training, just follow the instructions in the **Train** section. You can directly train on these datasets after editing `train.py`.
   
3. [ETH]() and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data):
   
   These two datasets contain annotated trajectories of the real-world pedestrians with a variety of social interaction scenarios. Each frame in both datasets includes top-view images and 2D locations of each person in the world coordinate.

   ETH dataset consists of two unique scenes, ETH and Hotel, while UCY dataset contains 3 unique scenes, Zara01, Zara02, and Univ. Modified version of these datasets are also uploaded. Edit `train.py` in the same way as the **SDD** part and then you can train on these datasets.

If you would like to develope your work on any of our modified version of datasets above, please be sure to attribute not only our work but also the original datasets and follow the licenses of them.

## Performance
We compare the performance of our model against the various baselines on two heterogeneous datasets: Waymo and SDD, as well as on two commonly-used homogeneous datasets, ETH and UCY.

### Evaluation on SDD in terms of ADE and FDE
 Models | Lin | S-LSTM | S-GAN-P | DESIRE | SoPhie | AEE-GAN (Ours) | 
 -|:-:|:-:|:-:|:-:|:-:|:-:|
 **ADE / FDE**|37.11 / 63.51|31.19 / 56.98|28.31 / 42.63|19.25 / 34.05|16.27 / 29.38|12.45 / 14.00

### Baslines on Homogeneous Datasets
Datasets \ Models |S-LSTM|S-GAN-P|SoPhie|S-Ways|S-STGCNN|AEE-GAN (Ours)|
-|:-:|:-:|:-:|:-:|:-:|:-:|
ETH|1.09 / 2.35|0.77 / 1.38|0.70 / 1.43|0.39 / 0.64|0.62 / 1.07|0.32 / 0.44|
Hotel|0.79 / 1.76|0.44 / 0.89|0.76 / 1.67|0.39 / 0.66|0.41 / 0.51|0.19 / 0.23|
Univ|0.67 / 1.40|0.75 / 1.50|0.54 / 1.24|0.55 / 1.31|0.62 / 1.07|0.37 / 0.56|
ZARA1|0.47 / 1.00|0.35 / 0.69|0.30 / 0.63|0.44 / 0.64|0.40 / 0.61|0.24 / 0.33|
ZARA2|0.56 / 1.17|0.36 / 0.72|0.38 / 0.78|0.51 / 0.92|0.31 / 0.49|0.26 / 0.25|
**AVG**|0.71 / 1.53|0.53 / 1.02|0.53 / 1.15|0.45 / 0.83|0.48 / 0.73|0.27 / 0.36|

### Ablation studies on SDD and Waymo Dataset
Datasets|_Ho_|_So_|AEE-GAN (Ours)|
-|:-:|:-:|:-:|
Waymo|5.31 / 9.49|3.74 / 7.19|3.24 / 5.84|
SDD|15.88 / 27.67|13.83 / 21.03|12.46 / 14.01|
-|-|-|-|
ETH|0.35 / 0.53|0.34 / 0.43|0.32 / 0.44|
Hotel|0.30 / 0.36|0.22 / 0.23|0.19 / 0.23|
Univ|0.46 / 0.74|0.51 / 0.96|0.37 / 0.56|
ZARA1|0.28 / 0.47|0.25 / 0.34|0.24 / 0.33|
ZARA2|0.27 / 0.46|0.28 / 0.47|0.26 / 0.25|

## Citation
If you want to use this work for your research, please cite our paper.

```
@inproceedings{,
  title={Trajectory Prediction in Heterogeneous Environment via Attended Ecology Embedding},
  author={},
  booktitle={},
  year={2020}
}
```
