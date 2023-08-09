import torch as torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import kornia.geometry.transform as KT
import random

from .LocalGaussianBlur import *
from .noise import *
from .configs import *
from .extras import *

# --- Extra Functions

def misalign_channels(cloud):
    return cloud

def cloud_hue(image, cloud, scale=1.0):
    """
        Mixes the pure white base of a cloud with mean color of the underlying image
        
        scale (float) controls how 'white' the end result is (the lower the value, the more color the cloud will have)
    """
    # Mean Cloud-Free Color
    mean_color = image.mean((-2,-1))
    ambience = torch.ones_like(image)
    
    mask=(cloud*scale).clip(0,1)
    
    # Safety against pitch black
    for b_idx in range(mean_color.shape[0]):
        if all(mean_color[b_idx])==0:
            mean_color[b_idx]=torch.ones_like(mean_color[b_idx]) # prevent mixing with pitch black
        
    color_vector=mean_color/mean_color.mean(1,keepdim=True)
    color_vector/=color_vector.max(1,keepdim=True)[0] # ensure that no value exceeds 1.0        
    return ambience*mask + color_vector.unsqueeze(-1).unsqueeze(-1)*ambience*(1-mask)
    

# --- Mixing Methods

def mix(input, cloud, shadow=None, channel_magnitude=None, blur_scaling=2.0, cloud_color=True, invert=False):
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
    if channel_magnitude is None:
        channel_magnitude=torch.ones(*input.shape[:-2],1,1,device=input.device)
    else:
        channel_magnitude=channel_magnitude.view(*input.shape[:-2],1,1)
    
    if shadow is not None:
        # reuse the same function, with shadow mask as 'cloud' mask (and inverting the 
        input=mix(input, shadow, blur_scaling=0.0, cloud_color=False, invert=not invert)
    
    # blurring background (optional)
    if blur_scaling != 0.0:
        modulator = cloud.mean(1) # average cloud thickness
        input = local_gaussian_blur(input,
                                    blur_scaling*modulator)
        
    # mix the cloud
    if invert:
        output = input * (1-cloud.clip(0,1))
    else:
        # use max_lvl to multiply the resulting cloud base        
        max_lvl=cloud.max() if cloud.max()>1.0 else 1.0
        cloud_base = torch.ones_like(input) if not cloud_color else cloud_hue(input, cloud)
        cloud_base = channel_magnitude*cloud_base
        output = input*(1-cloud/max_lvl) + max_lvl*cloud_base*(cloud/max_lvl)
        
    return output

class CloudGenerator(torch.nn.Module):
    
    """ Wrapper object for the add_cloud() and add_cloud_and_shadow() methods.
        It stores the parameters to these functions in a config dictionary
    
    
    """
    
    def __init__(self,
                 config,
                 cloud_p=1.0,
                 shadow_p=1.0
                ):
        super().__init__()
        
        self.cloud_p=cloud_p
        self.shadow_p=shadow_p        
        
        if isinstance(config,dict):
            self.config=[config] # put into a list if it's a single config
        else:
            self.config=config
            
    def choose_config(self):
        return random.choice(self.config)
    
    def segmentation_mask(self,*args,**kwargs):
        # wraps for segmentation mask for external use
        return segmentation_mask(*args,**kwargs)
        
    def forward(self,img,*args,return_cloud=False,**kwargs):
        # decide which config from the list (if multiple)
        used_config=self.choose_config()
        
        # decide what to simulate
        do_cloud=random.random()<=self.cloud_p
        do_shadow=random.random()<=self.shadow_p
        
        # synth both cloud and shadow
        if do_shadow and do_cloud:           
            out=add_cloud_and_shadow(img,*args,**kwargs,**used_config, return_cloud=return_cloud)
            
            if return_cloud:
                out,cloud,shadow=out
            else:
                cloud,shadow=None,None
                
        # synth only cloud
        elif do_cloud:
            out=add_cloud(img,*args,**kwargs,**used_config,return_cloud=return_cloud)   
            
            if return_cloud:
                out,cloud=out
                shadow=torch.zeros_like(out)
            else:
                cloud,shadow=None,None
        # no additions
        else:            
            out=torch.from_numpy(img) if not torch.is_tensor(img) else img
            cloud,shadow=torch.zeros_like(out),torch.zeros_like(out)
        
        # return format
        if return_cloud:
            return out,cloud,shadow
        else:
            return out
        
    def __or__(self, other):
        cfg1=self.config
        cfg2=other.config
        # inherit maximum probability of the two parents
        cloud_p=max([self.cloud_p, other.cloud_p])
        shadow_p=max([self.shadow_p, other.shadow_p])
        return CloudGenerator(config=cfg1+cfg2,
                              cloud_p=cloud_p,
                              shadow_p=shadow_p)
    
    def __str__(self):
        N=len(self.config)
        
        return ("CloudGenerator(cloud_p={:.2f},shadow_p={:.2f},{} config(s))").format(self.cloud_p,self.shadow_p,N)
    
    def __repr__(self):
        N=len(self.config)
        
        config_desc=""
        for c in self.config:
            config_desc+="\n{"
            for key in c:
                config_desc+="\t{}: {}\n".format(key,c[key])
            config_desc+="}"
        
        return ("CloudGenerator(cloud_p={:.2f},shadow_p={:.2f},\n{} config(s):{})").format(self.cloud_p,self.shadow_p,N,config_desc)

def add_cloud(input,
              max_lvl=(0.95,1.0),
              min_lvl=(0.0, 0.05),
              channel_magnitude=None,
              clear_threshold=0.0,
              noise_type = 'perlin',
              const_scale=True,
              decay_factor=1,
              locality_degree=1,
              invert=False,
              channel_magnitude_shift=0.05,
              channel_offset=2,
              blur_scaling=2.0,
              cloud_color=True,
              return_cloud=False
             ):
    """ Takes an input image of shape [batch, channels, height, width]        
        and returns a generated cloudy version of the input image
    
    Args:
        input (Tensor) : input image in shape [B,C,H,W]
    
        max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
        
        min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
        channel_magnitude (Tensor) : cloud magnitudes in each channel, shape [B,C,1,1]
        
        clear_threshold (float): An optional threshold for cutting off some part of the initial generated cloud mask
        
        noise_type (string: 'perlin', 'flex'): Method of noise generation (currently supported: 'perlin', 'flex')
        
        const_scale (bool): If True, the spatial frequencies of the cloud shape are scaled based on the image size (this makes the cloud preserve its appearance regardless of image resolution)
        
        decay_factor (float): decay factor that narrows the spectrum of the generated noise (higher values, such as 2.0 will reduce the amplitude of high spatial frequencies, yielding a 'blurry' cloud)
        
        locality degree (int): more local clouds shapes can be achieved by multiplying several random cloud shapes with each other (value of 1 disables this effect, and higher integers correspond to the number of multiplied masks)
        
        invert (bool) : for some applications, the cloud can be inverted to effectively decrease the level of reflected power (see thermal example in the notebook)
        
        channel_offset (int): optional offset that can randomly misalign spatially the individual cloud mask channels (by a value in range -channel_offset and +channel_offset)
        
        channel_magniutde_shift (float): optional offset from the reference cloud mask magnitude for individual channels, if non-zero, then each channel will have a cloud magnitude uniformly sampled from C+-channel_magnitude, where C is the reference cloud mask
        
        blur_scaling (float): Scaling factor for the variance of locally varying Gaussian blur (dependent on cloud thickness). Value of 0 will disable this feature.
        
        cloud_color (bool): If True, it will adjust the color of the cloud based on the mean color of the clear sky image
        
        return_cloud (bool): If True, it will return a channel-wise cloud mask of shape [height, width, channels] along with the cloudy image
        
    Returns:
    
        Tensor: Tensor containing a generated cloudy image (and a cloud mask if return_cloud == True)
  
    """  
    
    if not torch.is_tensor(input):
        input = torch.FloatTensor(input)
    
    while len(input.shape) < 4:
        input = input.unsqueeze(0)  
    
    b,c,h,w = input.shape
    device=input.device
    
    # --- Potential Sampling of Parameters (if provided as a range)
    min_lvl=torch.tensor(min_lvl, device=device)
    max_lvl=torch.tensor(max_lvl, device=device)
    
    if len(min_lvl.shape) != 0:
        min_lvl = min_lvl[0] +(min_lvl[1]-min_lvl[0])*torch.rand([b,1,1,1], device=device)
        
    # max_lvl is dependent on min_lvl (cannot be less than min_lvl)
    if len(max_lvl.shape) != 0:        
        max_floor=min_lvl+F.relu(max_lvl[0]-min_lvl)
        max_lvl = max_floor + (max_lvl[1]-max_floor)*torch.rand([b,1,1,1], device=device)
        
    # ensure max_lvl does not go below min_lvl
    max_lvl=min_lvl+F.relu(max_lvl-min_lvl)
        
    # clear_threshold
    if isinstance(clear_threshold, tuple) or isinstance(clear_threshold, list):
        clear_threshold = clear_threshold[0] +(clear_threshold[1]-clear_threshold[0])*torch.rand([b,1,1], device=device)
        
    # decay_factor
    if isinstance(decay_factor, tuple) or isinstance(decay_factor, list):
        decay_factor = float(decay_factor[0] +(decay_factor[1]-decay_factor[0])*torch.rand([1,1]))

    # locality_degree
    if isinstance(locality_degree, tuple) or isinstance(locality_degree, list):
        locality_degree = int(locality_degree[0]+torch.randint(1+locality_degree[1]-locality_degree[0],(1,1)))
    
    # --- End of Parameter Sampling
    locality_degree=max([1, int(locality_degree)])
    
    net_noise_shape=torch.ones((b,h,w),device=device)
    for idx in range(locality_degree):
        # generate noise shape
        if noise_type == 'perlin':
            noise_shape=generate_perlin(shape=(h,w), batch=b, device=device, const_scale=const_scale, decay_factor=decay_factor)     
        elif noise_type == 'flex':
            noise_shape = flex_noise(h,w, const_scale=const_scale, decay_factor=decay_factor)
        else:
            raise NotImplementedError

        noise_shape -= noise_shape.min()
        noise_shape /= noise_shape.max()
        
        net_noise_shape*=noise_shape
        
    # apply non-linearities and rescale
    net_noise_shape[net_noise_shape < clear_threshold] = 0.0
    net_noise_shape -= clear_threshold  
    net_noise_shape = net_noise_shape.clip(0,1)    
    if not net_noise_shape.max()==0:
        net_noise_shape /= net_noise_shape.max()

    # channel-wise mask
    cloud=(net_noise_shape.unsqueeze(1)*(max_lvl-min_lvl) + min_lvl).expand(b,c,h,w)
    
    # channel-wise thickness difference
    if channel_magnitude_shift != 0.0:
        channel_magnitude_shift=abs(channel_magnitude_shift)
        weights=channel_magnitude_shift*(2*torch.rand(c, device=device)-1)+1
        cloud=(weights[:,None,None]*cloud)
    
    # channel offset (optional)
    if channel_offset != 0:
        offsets = torch.randint(-channel_offset, channel_offset+1, (2,c))
        
        crop_val = offsets.max().abs()
        if crop_val != 0:
            for ch in range(cloud.shape[1]):
                cloud[:,ch] = torch.roll(cloud[:,ch], offsets[0,ch].item(),dims=-2)
                cloud[:,ch] = torch.roll(cloud[:,ch], offsets[1,ch].item(),dims=-1)                    

                cloud = KT.resize(cloud[:,:,crop_val:-crop_val-1, crop_val:-crop_val-1],
                                  (h,w),
                                  interpolation='bilinear',
                                  align_corners=True)     
    
    # transparency between 0 and 1
    cloud=cloud.clip(0,1)
    
    if channel_magnitude is None:
        channel_magnitude=torch.ones(*input.shape[:-2],1,1,device=input.device)
                
    output = mix(input, cloud, channel_magnitude=channel_magnitude, blur_scaling=blur_scaling, cloud_color=cloud_color, invert=invert)
    
    if not return_cloud:
        return output
    else:
        return output, cloud# if not invert else 1-cloud

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
        and returns a generated cloudy version of the input image, with additional shadows added to the ground image
    
    Args:
        
        input (Tensor) : input image in shape [B,C,H,W]
    
        max_lvl (float or tuple of floats): Indicates the maximum strength of the cloud (1.0 means that some pixels will be fully non-transparent)
        
        min_lvl (float or tuple of floats): Indicates the minimum strength of the cloud (0.0 means that some pixels will have no cloud)
        channel_magnitude (Tensor) : cloud magnitudes in each channel, shape [B,C,1,1]
        
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
    if isinstance(locality_degree,int):
        shadow_locality_degree=locality_degree-1
    else:
        shadow_locality_degree=[i-1 for i in locality_degree]
        
    # but don't add shadows if cloud level 'floor' is above 0...
    if isinstance(min_lvl,list) or isinstance(min_lvl,tuple):
        if min_lvl[0] > 0.0:
            shadow_max_lvl=0.0
    else:
        if min_lvl > 0.0:
            shadow_max_lvl=0.0
        
    input, shadow_mask = add_cloud(input,
                                   max_lvl=shadow_max_lvl,
                                   min_lvl=0.0,
                                   clear_threshold=0.4,
                                   noise_type = 'perlin',
                                   const_scale=const_scale,
                                   decay_factor=1.5, # Suppress HF detail
                                   locality_degree=shadow_locality_degree, # make similar locality as cloud (-1 works well because it's lower frequency)
                                   invert=True, # Invert Color for shadow
                                   channel_offset=0, # Cloud SFX disabled
                                   channel_magnitude_shift=0.0, # Cloud SFX disable
                                   blur_scaling=0.0, # Cloud SFX disabled
                                   cloud_color=False, # Cloud SFX disabled
                                   return_cloud=True
             )
    
    # 2. Add Cloud
    input, cloud_mask = add_cloud(input,
                                  max_lvl=max_lvl,
                                  min_lvl=min_lvl,
                                  channel_magnitude=channel_magnitude,
                                  clear_threshold=clear_threshold,
                                  noise_type=noise_type,
                                  const_scale=const_scale,
                                  decay_factor=decay_factor,
                                  locality_degree=locality_degree,
                                  invert=False,
                                  channel_offset=channel_offset,
                                  channel_magnitude_shift=channel_magnitude_shift,
                                  blur_scaling=blur_scaling,
                                  cloud_color=cloud_color,
                                  return_cloud=True
                                 )
    
    if not return_cloud:
        return input
    else:
        return input, cloud_mask, shadow_mask