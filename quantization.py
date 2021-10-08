import torch

from slender.prune.vanilla import prune_vanilla_elementwise
from slender.quantize.linear import quantize_linear, quantize_linear_fix_zeros
from slender.quantize.kmeans import quantize_k_means, quantize_k_means_fix_zeros
from slender.quantize.fixed_point import quantize_fixed_point
from slender.quantize.quantizer import Quantizer


def test_quantize_linear():
    param = torch.rand(128, 64, 3, 3) - 0.5
    codebook = quantize_linear(param, k=16)
    assert codebook['cluster_centers_'].numel() == 16
    centers_ = codebook['cluster_centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
