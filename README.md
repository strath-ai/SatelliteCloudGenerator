# :cloud: Satellite Cloud Generator
[![MDPI](https://img.shields.io/badge/Open_Access-MDPI-green)](https://www.mdpi.com/2072-4292/15/17/4138) [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cidcom/SatelliteCloudGenerator/blob/main/01c_Usage_Examples_Colab.ipynb) [![Zenodo](https://zenodo.org/badge/532972529.svg)](https://zenodo.org/badge/latestdoi/532972529)
[![[YouTube]](https://img.shields.io/badge/YouTube-%23FF0000.svg?style=flat&logo=youtube&logoColor=white)](https://youtu.be/RgAF2Y4O9zA)

:star: **NEW:** This tool has been featured in the 📺 first episode of the [**satellite-image-deep-learning podcast**](https://youtu.be/RgAF2Y4O9zA)! :star: 

---

A PyTorch-based tool for simulating clouds in satellite images.

This tool allows for generating artificial clouds in an image using structural noise, such as Perlin noise; intended for applications where pairs of clear-sky and cloudy images are required or useful.
For example, it can be used to **generate training data** for tasks such as **cloud detection** or **cloud removal**, or simply as a method of **augmentation** of satellite image data for other tasks.

The images must be in shape `(channel, height, width)` or `(batch, channel, height, width)` and are also returned in that format.

---

### Open Access Journal
This tool is accompanied by the **open access publication** at https://www.mdpi.com/2072-4292/15/17/4138.

If you found this tool useful, please cite accordingly:
```bibtex
@Article{rs15174138,
  author = {Czerkawski, Mikolaj and Atkinson, Robert and Michie, Craig and Tachtatzis, Christos},
  title = {SatelliteCloudGenerator: Controllable Cloud and Shadow Synthesis for Multi-Spectral Optical Satellite Images},
  journal = {Remote Sensing},
  volume = {15},
  year = {2023},
  number = {17},
  article-number = {4138},
  url = {https://www.mdpi.com/2072-4292/15/17/4138},
  issn = {2072-4292},
  doi = {10.3390/rs15174138}
}
```

### Installation
```bash
pip install git+https://github.com/strath-ai/SatelliteCloudGenerator
```

and then import:
```python
import satellite_cloud_generator as scg

cloudy_img = scg.add_cloud_and_shadow(clear_img)
```

## :gear: Usage
Basic usage, takes a `clear` image and returns a `cloudy` version along with a corresponding channel-specific transparency `mask`:
```python
cloudy, mask = scg.add_cloud(clear,
                             min_lvl=0.0,
                             max_lvl=1.0
                         )
```
...resulting in the following:

![Basic Example](imgs/thick_cloud.png)

The `min_lvl` and `max_lvl` control the range of values of the transparency `mask`.

### Generator Module
You can also use a `CloudGenerator` object that binds a specific configuration (or a set of configurations) with the wrapped generation methods:
```python
my_gen=scg.CloudGenerator(scg.WIDE_CONFIG,cloud_p=1.0,shadow_p=0.5)
my_gen(my_image) # will act just like add_cloud_and_shadow() but will preserve the same configuration!
```

## Selected Features (There's more! Scroll down for full list)
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
> :warning: The blur operation significantly increases memory footprint (caused by the internal `unfold` operation).

![Blur](imgs/back_blur.png)

## Summary of Parameters
```python
def add_cloud_and_shadow(input,
                         max_lvl=(0.95,1.0),
                         min_lvl=(0.0, 0.05),
                         channel_magnitude=None,
                         shadow_max_lvl=[0.3,0.6],
                         clear_threshold=0.0,
                         noise_type = 'perlin',
                         const_scale=True,
                         decay_factor=1,
                         locality_degree=1,
                         channel_offset=2,
                         channel_magnitude_shift=0.05,
                         blur_scaling=2.0,
                         cloud_color=True,
                         return_cloud=False
                        ):
    """ Takes an input image of shape [batch,channels,height, width]        
        and returns a generated cloudy version of the input image, with additional shadows added to the ground image"""
```

| Argument | Description | Default value |
|:---:|:---|:---| 
| input (Tensor) | input image in shape [B,C,H,W]| |
|max_lvl (float or tuple of floats) | Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)|`(0.95,1.0)`|
|min_lvl (float or tuple of floats)| Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)|`(0.0, 0.05)`|
|channel_magnitude (Tensor) | (optional) cloud magnitudes in each channel, shape [B,C,1,1]|`None`|
|clear_threshold (float)|An optional threshold for cutting off some part of the initial generated cloud mask|`0.0`|
|shadow_max_lvl (float)|Indicates the maximum strength of the cloud (1.0 means that some pixels will be completely black)|`[0.3,0.6]`|
|noise_type (string: 'perlin', 'flex')|Method of noise generation (currently supported: 'perlin', 'flex')|`'perlin'`|
|const_scale (bool)|If True, the spatial frequencies of the cloud/shadow shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)|`True`|
|decay_factor (float)|decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)|`1`|
|locality degree (int)|more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)|`1`|  
|channel_offset (int)|optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)|`2`| 
|blur_scaling (float)|Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.|`2.0`|   
|cloud_color (bool)|If True, it will adjust the color of the cloud based on the mean color of the clear sky image|`True`|
|return_cloud (bool)|If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image|`True`|
