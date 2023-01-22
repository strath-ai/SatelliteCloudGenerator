import torch
import numpy as np

# Noise Generation Methods

def output_transform(x):
    # normalize to max magnitude of 1
    x /= max([x.max(), -x.min()])
    # pass through tanh to ensure predefined range
    return (4*x).tanh()

# --- Perlin Methods
def interp(t):
    return 3 * t**2 - 2 * t ** 3

def perlin(width, height, scale=10, batch=1, device=None):
    # based on https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
    
    gx, gy = torch.randn(2, batch, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1, device=device)[:-1, None]
    ys = torch.linspace(0, 1, scale + 1, device=device)[None, :-1]

    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    
    dots = 0
    dots += wx * wy * (gx[:,:-1, :-1] * xs + gy[:,:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[:,1:, :-1] * (1 - xs) + gy[:,1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:,:-1, 1:] * xs - gy[:,:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[:,1:, 1:] * (1 - xs) - gy[:,1:, 1:] * (1 - ys))

    return dots.permute(0, 1, 3, 2, 4).contiguous().view(batch, width * scale, height * scale)

def generate_perlin(scales=None, shape=(256,256), batch=1, device='cpu', weights=None, const_scale=True, decay_factor=1):
    # Set Up Scales
    if scales is None:
        up_lim = max([2, int(np.log2(min(shape)))-1])        
        
        scales = [2**i for i in range(2,up_lim)]
        # proportional to image size, if const_scale is preserved
        if const_scale:
            f = int(2**np.floor(np.log2(0.25*max(shape)/max(scales))))
            scales = [el*f for el in scales]

    if weights is None:
        weights = [el**decay_factor for el in scales]
    # Round shape to nearest power of 2 
    big_shape = [int(2**(np.ceil(np.log2(i)))) for i in shape]
    out = torch.zeros([batch,*shape], device=device)
    for scale, weight in zip(scales, weights):
        out += weight*perlin(int(big_shape[0]/scale), int(big_shape[1]/scale), scale=scale, batch=batch, device=device)[...,:shape[0],:shape[1]]

    return output_transform(out)

# --- Flexible Noise Filtering Methods
def default_weight(input, const_scale=True, decay_factor=1):
    
    # scaling factor        
    if const_scale:
        factor_multiplier=0.32*max(input.shape)
    else:
        factor_multiplier=64
    
    if isinstance(decay_factor, list) or isinstance(decay_factor, tuple):
        ret=0
        decay_factor=sorted(decay_factor)[::-1]
        for f in decay_factor:
            ret+=torch.exp(-factor_multiplier*f*input)/(decay_factor[0]/f)
    else:
        ret = torch.exp(-factor_multiplier*decay_factor*input)
    return ret

def flex_noise(width, height, spectral_weight=default_weight, const_scale=False, decay_factor=1):
    raise NotImplemented # This needs to be converted to work with batches, and also generate full on the requested device
    # Source Noise
    x = torch.rand(width,height) - 0.5    
    x_f = torch.fft.rfft2(x)
    
    # Weight Space (value proportional to euclidean distance from DC)
    x_space = torch.abs(torch.arange(-x_f.shape[0]/2,
                                     x_f.shape[0]/2,
                                     1.0))*(x.shape[1]/x.shape[0]) # scaling to preserve 'aspect'
    y_space = torch.abs(torch.arange(0,
                                     x_f.shape[1],
                                     1.0))    
    X_grid,Y_grid = torch.meshgrid(x_space,y_space)

    W_space = (X_grid**2+Y_grid**2)**0.5
    W_space /= W_space.max()    

    # Modulation of Weight
    M = spectral_weight(W_space, const_scale=const_scale, decay_factor=decay_factor)
    
    # Application to Noise Spectrum
    return output_transform(torch.fft.irfft2(torch.fft.fftshift(M,0)*x_f))