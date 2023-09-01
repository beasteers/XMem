

# default configuration
DEFAULT_CONFIG = {
    'top_k': 30,
    'num_prototypes': 128,
    # memory update frequency
    'mem_every': 5,
    'deep_update_every': -1,
    # long-term flags
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    # memory consolidation options
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
    # these are used for very small masks e.g. toothpicks
    'dilate_size_threshold': 0,
    'dilation_kernel_size': 5,
    # mask matching options
    'min_iou': 0.4,
    'allow_create': True,
    # these are useful for uncertain mask detections
    'tentative_frames': 0,
    'max_age': 0,
}

def get_config(config):
    return {**DEFAULT_CONFIG, **(config or {})}