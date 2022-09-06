# :cloud: Satellite Cloud Generator
A PyTorch-based tool for simulating clouds and shadows in satellite images.

This tool allows for generating artificial clouds in an image, intended for applications, where pairs of clear-sky and cloudy images are required or useful.
For example, it can be used to **generating training data** for tasks such as **cloud detection** or **cloud removal**, or simply as a method of **augmention** of satellite image data for other tasks.

The images must be in shape `(height, width, channel)` and are also returned in that format.

### Requirements
```bash
torch>=1.10.0
torchvision
numpy
imageio
```

## :gear: Usage
Basic usage:
```python
cloudy = add_cloud(clear,
                   min_lvl=0.0,
                   max_lvl=1.0,
                   cloud_color=True,
                   channel_offset=2,
                   blur_scaling=4.0
                  )
```
...resulting in the following:

![Basic Example](imgs/thick_cloud.png)

## Features
Apart from synthesizing a random cloud transparency mask, the tool provides several features to make the appearance of the clouds more realistic.

### 1. Cloud Color
The `cloud_color` parameter set to `True` will adjust the color of the base added cloud based on the mean color of the clear ground image.

![Cloud Color](imgs/cloud_color.png)
---
### 2. Channel Offset
Spatial offsets between individual cloud image channels can be achieved by setting `channel_offset` to a positive integer value.

![Channel Offset](imgs/channel_offset.png)
---
### 3. Blur-Under-the-Cloud
Blurring of the ground image based on the cloud thickness can be achieved by adjusting the `blur_scaling` parameter (with `0.0` disabling the effect).

![Blur](imgs/back_blur.png)

---

The majority of features are based on the following paper:
```bibtex
@inproceedings{Lee2019,
author={Lee, Kyu-Yul and Sim, Jae-Young},
booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
title={Cloud Removal of Satellite Images Using Convolutional Neural Network With Reliable Cloudy Image Synthesis Model}, 
year={2019},
pages={3581-3585},
doi={10.1109/ICIP.2019.8803666}}
```
