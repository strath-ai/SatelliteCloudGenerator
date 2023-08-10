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
    
    
    Expected input shape for cloud and shadow: (H,W) or (C,H,W) or (B,C,H,W)
    """
    
    while len(cloud.shape)<4:
        cloud.unsqueeze_(0)
    
    if shadow is None:
        shadow=torch.zeros_like(cloud)
    while len(shadow.shape)<4:
        shadow.unsqueeze_(0)
        
    # cloud and shadow are (B,C,H,W) by now
    b,c,h,w=cloud.shape
        
    seg_mask=torch.zeros(b,h,w)
    
    # get binary representations
    thick_cloud_b=1.0*(cloud.mean(-3)>=thin_range[1])
    thin_cloud_b=1.0*(cloud.mean(-3)<thin_range[1])*(cloud.mean(-3)>=thin_range[0])*(1.0-thick_cloud_b)
    shadow_b=1.0*(shadow.mean(-3)>shadow_threshold)*(1.0-thick_cloud_b)*(1.0-thin_cloud_b)

    if thin_range[0]==thin_range[1]:
        seg_mask=thick_cloud_b + 2*shadow_b
    else:
        seg_mask=thick_cloud_b + 2*thin_cloud_b + 3*shadow_b
    
    return seg_mask