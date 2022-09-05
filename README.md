# :cloud: satellite-cloud-simulator
A PyTorch-based tool for simulating clouds and shadows in satellite images.

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
Apart from synthesizing a random cloud transparency mask, the tool provides several features to make the appearance of the clouds more realistic:

### 1. Cloud Color
The `cloud_color` parameter set to `True` will adjust the color of the base added cloud based on the mean color of the clear ground image.

![Cloud COlor](imgs/cloud_color.png)
---
### 2. Channel Offset
Spatial offsets between individual cloud image channels can be achieved by setting `channel_offset` to a positive integer value.

![Channel Offset](imgs/channel_offset.png)
---
### 3. Blur-Under-the-Cloud
Blurring of the ground image based on the cloud thickness can be achieved by adjusting the `blur_scaling` parameter (with `0.0` disabling the effect).

![Blur](imgs/back_blur.png)
