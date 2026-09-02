import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Set seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For MPS (Metal Performance Shaders) which seems to be used in this project
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Ensure deterministic behavior in some torch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms if available
    try:
        torch.use_deterministic_algorithms(True)
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except Exception:
        pass
