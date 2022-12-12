import torch as torch
import numpy as np
from matplotlib import pyplot as plt
import kornia.geometry.transform as KT

from .LocalGaussianBlur import *
from .noise import *

# --- Extra Functions
def misalign_channels(cloud):
    return cloud

def cloud_hue(image, cloud, scale=1.0):
    """
        Mixes the pure white base of a cloud with mean color of the underlying image
        
        scale (float) controls how 'white' the end result is (the lower the value, the more color the cloud will have)
    """
    # Mean Cloud-Free Color
    mean_color = image.mean((0,1))
    ambience = torch.ones_like(image)
    
    mask=(cloud*scale).clip(0,1)
    
    if all(mean_color==0):
        return ambience # prevent mixing with pitch black
    else:
        color_vector=(mean_color/mean_color.mean())
        color_vector/=color_vector.max() # ensure that no value exceeds 1.0
        return ambience*mask + color_vector*ambience*(1-mask)
    

# --- Mixing Methods

def mix(input, cloud, shadow=None, blur_scaling=2.0, cloud_color=True, invert=False):
    """ Mixing Operation for an input image and a cloud
    
        Args:
            input (Tensor) : input image [height, width, channels]  

            cloud (Tensor) : cloud of the same shape [height, width, channels] 

            blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.

            cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
            
            invert (bool) : for some applications, the cloud can be inverted to effectively decrease the level of reflected power (see thermal example in the notebook)
            
        Returns:
    
            Tensor: Tensor containing a mixed image
    
    """
    if shadow is not None:
        # reuse the same function, with shadow mask as 'cloud' mask (and inverting the 
        input=mix(input, shadow, blur_scaling=0.0, cloud_color=False, invert=not invert)
    
    # blurring background (optional)
    if blur_scaling != 0.0:
        modulator = cloud.permute(2,0,1).mean(0)
        input = local_gaussian_blur(input.permute(2,0,1),
                                    blur_scaling*modulator)[0].permute(1,2,0)
        
    # mix the cloud
    if invert:
        output = input * (1-cloud)
    else:
        cloud_base = torch.ones_like(input) if not cloud_color else cloud_hue(input, cloud)
        output = input * (1-cloud) + cloud_base * (cloud)
        
    return output

class CloudGenerator(torch.nn.Module):
    
    """ Wrapper object for 
    
    
    """
    
    def __init__(self, config):
        super().__init__()
        self.config=config
        
    def forward(self,img):
        return add_cloud(img,**self.config)
    

def add_cloud(input,
              max_lvl=(0.95,1.0),
              min_lvl=(0.0, 0.05),
              clear_threshold=0.0,
              noise_type = 'perlin',
              const_scale=True,
              decay_factor=1,
              locality_degree=1,
              invert=False,
              channel_offset=2,
              blur_scaling=2.0,
              cloud_color=True,
              return_cloud=False
             ):
    """ Takes an input image of shape [height, width, channels]        
        and returns a generated cloudy version of the input image
    
    Args:
        max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
        
        min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
        clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
        
        noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
        
        const_scale (bool): If True, the spatial frequencies of the cloud shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)
        
        decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
        
        locality degree (int): more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)
        
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
        
    # max_lvl is dependent on min_lvl (cannot be less than min_lvl)
    if isinstance(max_lvl, tuple) or isinstance(max_lvl, list):
        max_floor=max([max_lvl[0],min_lvl])
        max_lvl = max_floor + (max_lvl[1]-max_floor)*torch.rand(1).item()
      
    locality_degree=max([1, int(locality_degree)])
    
    net_noise_shape=torch.ones((h,w))
    for idx in range(locality_degree):
        # generate noise shape
        if noise_type == 'perlin':
            noise_shape = generate_perlin(shape=(h,w), const_scale=const_scale, decay_factor=decay_factor).numpy()              
        elif noise_type == 'flex':
            noise_shape = flex_noise(h,w, const_scale=const_scale, decay_factor=decay_factor).numpy()
        else:
            raise NotImplementedError

        noise_shape -= noise_shape.min()
        noise_shape /= noise_shape.max()
        
        net_noise_shape*=noise_shape

    noise_shape = torch.FloatTensor(net_noise_shape)
        
    # apply non-linearities
    noise_shape[noise_shape < clear_threshold] = 0.0
    noise_shape -= clear_threshold  
    noise_shape = noise_shape.clip(0,1)
    noise_shape /= noise_shape.max()

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
    
    output = mix(input, cloud, blur_scaling=blur_scaling, cloud_color=cloud_color, invert=invert)
    
    if not return_cloud:
        return output
    else:
        return output, cloud# if not invert else 1-cloud

def add_cloud_and_shadow(input,
                         max_lvl=(0.95,1.0),
                         min_lvl=(0.0, 0.05),
                         shadow_max_lvl=0.3,
                         clear_threshold=0.0,
                         noise_type = 'perlin',
                         const_scale=True,
                         decay_factor=1,
                         locality_degree=1,
                         channel_offset=2,
                         blur_scaling=2.0,
                         cloud_color=True,
                         return_cloud=False
                        ):
    """ Takes an input image of shape [height, width, channels]        
        and returns a generated cloudy version of the input image, with additional shadows added to the ground image
    
    Args:
        max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
        
        min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
        clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
        
        shadow_max_lvl (float): Indicates the maximum strength of the cloud (1.0 means that some pixels will be completely black)
        
        noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
        
        const_scale (bool): If True, the spatial frequencies of the cloud/shadow shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)
        
        decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
        
        locality degree (int): more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)
        
        channel_offset (int): optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)
        
        blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.
        
        cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
        
        return_cloud (bool): If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image
        
    Returns:
    
        Tensor: Tensor containing a generated cloudy image (and a cloud mask if return_cloud == True)
  
    """  
    
    # 1. Add Shadows
    input, shadow_mask = add_cloud(input,
                                   max_lvl=shadow_max_lvl,
                                   min_lvl=0.0,
                                   clear_threshold=0.5,
                                   noise_type = 'perlin',
                                   const_scale=const_scale,
                                   decay_factor=1.5, # Suppress HF detail
                                   locality_degree=locality_degree-1, # make similar locality as cloud (-1 works well because it's lower frequency)
                                   invert=True, # Invert Color for shadow
                                   channel_offset=0, # Cloud SFX disabled
                                   blur_scaling=0.0, # Cloud SFX disabled
                                   cloud_color=False, # Cloud SFX disabled
                                   return_cloud=True
             )
    
    # 2. Add Cloud
    input, cloud_mask = add_cloud(input,
                                  max_lvl=max_lvl,
                                  min_lvl=min_lvl,
                                  clear_threshold=clear_threshold,
                                  noise_type=noise_type,
                                  const_scale=const_scale,
                                  decay_factor=decay_factor,
                                  locality_degree=locality_degree,
                                  invert=False,
                                  channel_offset=channel_offset,
                                  blur_scaling=blur_scaling,
                                  cloud_color=cloud_color,
                                  return_cloud=True
                                 )
    
    if not return_cloud:
        return input
    else:
        return input, cloud_mask, shadow_mask