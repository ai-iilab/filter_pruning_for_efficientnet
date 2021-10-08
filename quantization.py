import torch

from slender.prune.vanilla import prune_vanilla_elementwise
from slender.quantize.linear import quantize_linear, quantize_linear_fix_zeros
from slender.quantize.kmeans import quantize_k_means, quantize_k_means_fix_zeros
from slender.quantize.fixed_point import quantize_fixed_point
from slender.quantize.quantizer import Quantizer

def test_quantize_k_means():
    param = torch.rand(128, 64, 3, 3) - 0.5
    codebook = quantize_k_means(param, k=16)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    param = torch.rand(128, 64, 3, 3)
    codebook = quantize_k_means(param, k=16, codebook=codebook,
                                update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_


def test_quantize_k_means_fix_zeros():
    param = torch.rand(128, 64, 3, 3) - 0.5
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_k_means_fix_zeros(param, k=16)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()
    codebook = quantize_k_means_fix_zeros(param, k=16, codebook=codebook,
                                          update_centers=True)
    assert codebook.cluster_centers_.numel() == 16
    centers_ = codebook.cluster_centers_.view(16).tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()

    
def test_quantized_fixed_point():
    param = torch.rand(128, 64, 3, 3) - 0.5
    mask = prune_vanilla_elementwise(sparsity=0.4, param=param)
    codebook = quantize_fixed_point(param, bit_length=8, bit_length_integer=1)
    assert codebook['cluster_centers_'].numel() == 2 ** 8
    centers_ = codebook['cluster_centers_'].tolist()
    vals = set(param.view(param.numel()).tolist())
    for v in vals:
        assert v in centers_
    assert param.masked_select(mask).eq(0).all()
