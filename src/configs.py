# wide
WIDE_CONFIG={'min_lvl': [0.0,0.8],
             'max_lvl': [0.0,1.0],
             'const_scale':True,
             'decay_factor':[0.8,1.2],
             'clear_threshold':[0.0,0.1],
             'locality_degree':[1,3],
             'cloud_color':True,
             'channel_offset':2,
             'blur_scaling':2
            }

# thick global
BIG_CONFIG={'min_lvl':0.0,
            'max_lvl':1.0,
            'const_scale':True,
            'decay_factor':1.0,
            'clear_threshold':[0.0,0.2],
            'locality_degree':1,
            'cloud_color':True,
            'channel_offset':2,
            'blur_scaling':2
           }

# thick local
LOCAL_CONFIG={'min_lvl':0.0,
              'max_lvl':1.0,
              'const_scale':True,
              'decay_factor':1.0,
              'clear_threshold':[0.0,0.2],
              'locality_degree':[2,4],
              'cloud_color':True,
              'channel_offset':2,
              'blur_scaling':2
             }

# thin cloud
THIN_CONFIG={'min_lvl':[0.0,0.1],
            'max_lvl':[0.4,0.7],
            'const_scale':True,
            'decay_factor':1.0,
            'clear_threshold':0.0,
            'locality_degree':[1,3],
            'cloud_color':True,
            'channel_offset':4,
            'blur_scaling':2
     }

# foggy
FOG_CONFIG={'min_lvl':[0.3,0.6],
            'max_lvl':[0.6,0.7],
            'const_scale':True,
            'decay_factor':1.0,
            'clear_threshold':0.0,
            'locality_degree':1,
            'cloud_color':True,
            'channel_offset':2,
            'blur_scaling':2
     }