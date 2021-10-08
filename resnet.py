import torch
import torch.nn as nn

class basicblock(nn.Module):
    """Basic Block for Resnet 18 and Resnet 34

    """

    #basicblock and bottleneck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    
