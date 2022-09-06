# :cloud: Satellite Cloud Generator
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cidcom/SatelliteCloudGenerator/blob/main/01c_Usage_Examples_Colab.ipynb) [![DOI](https://zenodo.org/badge/532972529.svg)](https://zenodo.org/badge/latestdoi/532972529)

A PyTorch-based tool for simulating clouds in satellite images.

This tool allows for generating artificial clouds in an image using structural noise, such as Perlin noise; intended for applications where pairs of clear-sky and cloudy images are required or useful.
For example, it can be used to **generate training data** for tasks such as **cloud detection** or **cloud removal**, or simply as a method of **augmentation** of satellite image data for other tasks.

The images must be in shape `(height, width, channel)` and are also returned in that format.

> If you found this tool useful, please cite accordingly:
```bibtex
@software{7053356,
  author       = {Mikolaj Czerkawski, Christos Tachtatzis},
  title        = {Satellite Cloud Generator},
  month        = sep,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7053356},
  url          = {https://doi.org/10.5281/zenodo.7053356}
}
```

### Requirements
```bash
torch>=1.10.0
torchvision
kornia
numpy
imageio
```

## :gear: Usage
Basic usage, takes a `clear` image and returns a `cloudy` version along with a corresponding channel-specific transparency `mask`:
```python
cloudy, mask = add_cloud(clear,
                         min_lvl=0.0,
                         max_lvl=1.0
                         )
```
...resulting in the following:

![Basic Example](imgs/thick_cloud.png)

The `min_lvl` and `max_lvl` control the range of values of the transparency `mask`.

## Features
Apart from synthesizing a random cloud, the tool provides several additional features (switched on by default) to make the appearance of the clouds more realistic, inspired by [(Lee2019)](https://ieeexplore.ieee.org/document/8803666).

### 1. Cloud Color
The `cloud_color` setting adjusts the color of the base added cloud based on the mean color of the clear ground image. (Disable by passing `cloud_color=False`)

![Cloud Color](imgs/cloud_color.png)
---
### 2. Channel Offset
Spatial offsets between individual cloud image channels can be achieved by setting `channel_offset` to a positive integer value. (Disable by passing `channel_offset=0`)

![Channel Offset](imgs/channel_offset.png)
---
### 3. Blur-Under-the-Cloud
Blurring of the ground image based on the cloud thickness can be achieved by adjusting the `blur_scaling` parameter (with `0.0` disabling the effect). (Disable by passing `blur_scaling=0`)

![Blur](imgs/back_blur.png)

---
### Related Work
* (Dong2010) [Generation of Cloud Image Based on Perlin Noise ](https://ieeexplore.ieee.org/document/5694143)
* (Enomoto2017) [Filmy Cloud Removal on Satellite Imagery with Multispectral Conditional Generative Adversarial Nets](https://arxiv.org/abs/1710.04835)
* (Lee2019) [Cloud Removal of Satellite Images Using Convolutional Neural Network with Reliable Cloudy Image Synthesis Model](https://ieeexplore.ieee.org/document/8803666)
