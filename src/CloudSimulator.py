import torch as torch
import numpy as np
import kornia.geometry.transform as KT
import kornia.filters as KF
from matplotlib import pyplot as plt

from .LocalGaussianBlur import *
from .noise import *

# --- Extra Functions
def misalign_channels(cloud):
    return cloud

def cloud_hue(image, cloud):
    # Mean Cloud-Free Color
    mean_color = image.mean((0,1))
    ambience = torch.ones_like(image)
    
    return ambience*cloud + (mean_color/mean_color.mean())*ambience*(1-cloud)
    

# --- Mixing Methods

def add_cloud(input,
              max_lvl=(0.95,1.0),
              min_lvl=(0.0, 0.05),
              clear_threshold=0.0,
              noise_type = 'perlin',
              decay_factor=1,
              invert=False,
              channel_offset=2,
              blur_scaling=2.0,
              cloud_color=True,
              return_cloud=False
             ):
    """ Takes an input image of shape [height, width, channels]        
        and returns a generated cloudy version of the input image
    
    Args:
        max_lvl (float or tuple of floats): Indicates the maximum strength of the clear image (1.0 means that some pixels will be clear)
        
        min_lvl (float or tuple of floats): Indicates the minimum strength of the clear image (0.0 means that some pixels will have full cloud)
        clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
        
        noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
        
        decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
        
        invert (bool) : for some applications, the cloud can be inverted to effectively decrease the level of reflected power (see thermal example in the notebook)
        
        channel_offset (int): optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)
        
        blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.
        
        cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
        
        return_cloud (bool): If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image
        
    Returns:
    
        Tensor: Tensor containing a generated cloudy image (and a cloud mask if return_cloud == True)
  
    """  
    
    if not torch.is_tensor(input):
        input = torch.FloatTensor(input)
    
    if len(input.shape) == 2:
        input = input.unsqueeze(-1)
    
    h,w,c = input.shape[-3:]
    
    if isinstance(min_lvl, tuple) or isinstance(min_lvl, list):
        min_lvl = min_lvl[0] +(min_lvl[1]-min_lvl[0])*torch.rand(1).item()
        
    if isinstance(max_lvl, tuple) or isinstance(max_lvl, list):
        max_lvl = max_lvl[0] + (max_lvl[1]-max_lvl[0])*torch.rand(1).item()
      
    # generate noise shape
    if noise_type == 'perlin':
        noise_shape = generate_perlin(shape=(h,w),decay_factor=decay_factor).numpy()              
    elif noise_type == 'flex':
        noise_shape = flex_noise(h,w, decay_factor=decay_factor).numpy()
    else:
        raise NotImplementedError
        
    noise_shape -= noise_shape.min()
    noise_shape /= noise_shape.max()

    noise_shape = torch.FloatTensor(noise_shape)
        
    # apply non-linearities
    noise_shape[noise_shape < clear_threshold] = 0.0

    cloud = torch.stack(c*[1.0*noise_shape*(max_lvl-min_lvl) + min_lvl], 0)
    
    # channel offset (optional)
    if channel_offset != 0:
        offsets = torch.randint(-channel_offset, channel_offset+1, (2,c))
        
        crop_val = offsets.max().abs()
        if crop_val != 0:
            for ch in range(cloud.shape[0]):
                cloud[ch] = torch.roll(cloud[ch], offsets[0,ch].item(),dims=0)
                cloud[ch] = torch.roll(cloud[ch], offsets[1,ch].item(),dims=1)                    

                cloud = KT.resize(cloud[:,crop_val:-crop_val-1, crop_val:-crop_val-1],
                                  (h,w),
                                  interpolation='bilinear',
                                  align_corners=True)
    cloud = cloud.permute(1,2,0)
    
    # blurring background (optional)
    if blur_scaling != 0.0:
        modulator = 1-cloud.permute(2,0,1).mean(0)
        input = local_gaussian_blur(input.permute(2,0,1),
                                    blur_scaling*modulator)[0].permute(1,2,0)
        
    # mix the cloud
    if invert:
        output = input * (1 - cloud)
    else:
        cloud_base = torch.ones_like(input) if not cloud_color else cloud_hue(input, 1-cloud)
        output = input * cloud + cloud_base * (1-cloud)
    
    if not return_cloud:
        return output
    else:
        return output, 1.0-cloud if not invert else cloud
