import torch

# This file contains multiple approaches to extracting individual band magnitudes
# If no specific band strengths for the cloud are provided, the cloud component
# has approximately evenly distributed cloud strength across channels, obtained
# using cloud_hue() or torch.ones_like()

# most methods are based on the ratio between the clear region of a real reference image and the cloudy region

def mean_mag(reference,mask,mask_cloudy=None,clean=None):    
    """ Extract ratios of means
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).mean()
        clear_val=(i[mask_clean]).mean() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.mean() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def max_mag(reference,mask,mask_cloudy=None,clean=None):
    """ Extract ratios of max values
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).max()
        clear_val=(i[mask_clean]).max() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.median() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def median_mag(reference,mask,mask_cloudy=None,clean=None):
    """ Extract ratios of medians
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).median()
        clear_val=(i[mask_clean]).median() if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)

    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.median() if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base    

    return cloud_mag

def q_mag(reference,mask,mask_cloudy=None, clean=None,q=0.95,q2=None):
    """ Extract ratios of quantiles
    
        Args:
            reference (Tensor) : input reference image containing cloud [height, width, channels]  

            mask (Tensor) : mask, where 0.0 indicates a clear pixel (if mask_cloudy is provided, 1.0 indicates a clear pixel)
            
            mask_cloudy (Tensor) : optional mask, where 1.0 indicates cloud

            clean (Tensor) : optional image for multiplying the ratios by
            
            q (float) : quantile value used for the cloudy region
            
            q2 (float) : optional quantile value used for the clear region (if unspecifed, it is equal to q)
            
        Returns:
    
            Tensor: Tensor containing cloud magnitudes (or ratios if clean==None)
    
    """
    if mask_cloudy is None:
        mask_clean=mask.squeeze()==0.0 
        mask_cloudy=mask.squeeze()!=0.0 
    else:  
        mask_clean=mask
        mask_cloudy=mask_cloudy
    
    full_cloud=(mask_clean!=1.0).all()
    no_cloud=(mask_cloudy==0.0).all()
   
    if no_cloud:
        return None

    if q2 is None:
        q2=q
    
    # coef per band
    band_coefs=[]
    for idx in range(reference.shape[-3]):
        
        i=reference.index_select(-3,torch.tensor(idx,device=reference.device))

        cloud_val=(i[mask_cloudy]).quantile(q)
        clear_val=(i[mask_clean]).quantile(q2) if not full_cloud else 1

        band_coefs.append(cloud_val/clear_val)
        
    if clean is None:
        return band_coefs

    # cloud magnitude
    cloud_mag=torch.ones(clean.shape[:-2],device=clean.device)
    for idx in range(clean.shape[-3]):
        
        i=clean.index_select(-3,torch.tensor(idx,device=clean.device))
        base=i.quantile(q2) if not full_cloud else 1
        cloud_mag[...,idx]=band_coefs[idx]*base        

    return cloud_mag
