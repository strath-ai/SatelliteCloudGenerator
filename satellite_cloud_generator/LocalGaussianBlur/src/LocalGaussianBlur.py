import torch
from torch import Tensor

__all__ = [
    "LocalGaussianBlur",
    "local_gaussian_blur"
]

def gaussian_kernels(stds,size=11):
    """ Takes a series of std values of length N
        and integer size corresponding to kernel side length M
        and returns a set of gaussian kernels with those stds in a (N,M,M) tensor
    
    Args:
        stds (Tensor): Flat list tensor containing std values.
        size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Tensor containing a unique 2D Gaussian kernel for each value in the stds input.

    """  
    # 1. create input vector to the exponential function
    n = (torch.arange(0, size, device=stds.device) - (size - 1.0) / 2.0).unsqueeze(-1)
    var = 2*(stds**2).unsqueeze(-1) + 1e-8 # add constant for stability

    # 2. compute gaussian values with exponential
    kernel_1d = torch.exp((-n**2)/var.t()).permute(1,0)
    # 3. outer product in a batch
    kernel_2d = torch.bmm(kernel_1d.unsqueeze(2), kernel_1d.unsqueeze(1))
    # 4. normalize to unity sum
    kernel_2d /= kernel_2d.sum(dim=(-1,-2)).view(-1,1,1)
    
    return kernel_2d

def local_gaussian_blur(input, modulator, kernel_size=11):
    """Blurs image with dynamic Gaussian blur.
    
    Args:
        input (Tensor): The image to be blurred (C,H,W).
        modulator (Tensor): The modulating signal that determines the local value of kernel variance (H,W).
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Locally blurred version of the input image.

    """   
    
    if len(input.shape) < 4:
        input = input.unsqueeze(0)

    b,c,h,w = input.shape
    pad = int((kernel_size-1)/2)

    # 1. pad the input with replicated values
    inp_pad = torch.nn.functional.pad(input, pad=(pad,pad,pad,pad), mode='replicate')
    # 2. Create a Tensor of varying Gaussian Kernel
    kernels = gaussian_kernels(modulator.flatten()).view(b,-1,kernel_size,kernel_size)    
    #kernels_rgb = torch.stack(c*[kernels], 1)
    kernels_rgb=kernels.unsqueeze(1).expand(kernels.shape[0],c,*kernels.shape[1:])
    # 3. Unfold input
    inp_unf = torch.nn.functional.unfold(inp_pad, (kernel_size,kernel_size))  
    # 4. Multiply kernel with unfolded
    x1 = inp_unf.view(b,c,-1,h*w)
    x2 = kernels_rgb.view(b,c,h*w,-1).permute(0,1,3,2)#.unsqueeze(0)
    y = (x1*x2).sum(2)
    # 5. Fold and return
    return torch.nn.functional.fold(y,(h,w),(1,1))


class LocalGaussianBlur(torch.nn.Module):
    """Blurs image with dynamic Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [B, C, H, W] shape.
    
    Args:
        kernel_size (int): Size of the Gaussian kernel.

    Returns:
        Tensor: Gaussian blurred version of the input image.

    """     
    
    def __init__(self, kernel_size=11):
        super().__init__()
        self.kernel_size = kernel_size


    def forward(self, img: Tensor, modulator: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): image to be blurred.
            modulator (Tensor): signal modulating the kernel variance (shape H x W).

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        return local_gaussian_blur(img, modulator, kernel_size=self.kernel_size)


    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(kernel_size={self.kernel_size}"
        return s