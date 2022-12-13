import torch


def segmentation_mask(cloud,shadow=None,thin_range=(0.1,0.1),shadow_threshold=0.1):
    """ The following encoding method is used:
    0: Cleary Sky
    1: Cloud
    2: Shadow
    
    ...or, if thin_range contains a pair of *different numbers* (a,b):    
    0: Clear Sky
    1: Thick Cloud - cloud in range [a,b)
    2: Thin Cloud - cloud in range [b,+inf)
    3: Shadow
    
    
    Expected input shape for cloud and shadow: (H,W) or (H,W,C)
    """
    
    if len(cloud.shape)==2:
        cloud.unsqueeze_(-1)
    
    if shadow is None:
        shadow=torch.zeros_like(cloud)
    elif len(shadow.shape)==2:
        shadow.unsqueeze_(-1)
        
    # cloud and shadow are (H,W,C) by now
        
    seg_mask=torch.zeros(cloud.shape[:2])
    
    # get binary representations
    thick_cloud_b=1.0*(cloud.mean(-1)>=thin_range[1])
    thin_cloud_b=1.0*(cloud.mean(-1)<thin_range[1])*(cloud.mean(-1)>=thin_range[0])*(1.0-thick_cloud_b)
    shadow_b=1.0*(shadow.mean(-1)>shadow_threshold)*(1.0-thick_cloud_b)*(1.0-thin_cloud_b)

    if thin_range[0]==thin_range[1]:
        seg_mask=thick_cloud_b + 2*shadow_b
    else:
        seg_mask=thick_cloud_b + 2*thin_cloud_b + 3*shadow_b
    
    return seg_mask