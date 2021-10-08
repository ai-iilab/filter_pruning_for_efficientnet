import math
import torch

from slender.prune.vanilla import prune_vanilla_elementwise, prune_vanilla_kernelwise, \
    prune_vanilla_filterwise, VanillaPruner


def test_prune_vanilla_elementwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_elementwise(sparsity=0.3, param=param)
    assert mask.sum() == int(math.ceil(param.numel() * 0.3))
    assert param.masked_select(mask).eq(0).all()
    mask = prune_vanilla_elementwise(sparsity=0.7, param=param)
    assert mask.sum() == int(math.ceil(param.numel() * 0.7))
    assert param.masked_select(mask).eq(0).all()

def test_prune_vanilla_kernelwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_kernelwise(sparsity=0.5, param=param)
    mask_s = mask.view(64*128, -1).all(1).sum()
    assert mask_s == 32*128
    assert param.masked_select(mask).eq(0).all()
    
def test_prune_vanilla_filterwise():
    param = torch.rand(64, 128, 3, 3)
    mask = prune_vanilla_filterwise(sparsity=0.5, param=param)
    mask_s = mask.view(64, -1).all(1).sum()
    assert mask_s == 32
    assert param.masked_select(mask).eq(0).all()
