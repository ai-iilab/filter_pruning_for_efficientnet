import math
import torch

from slender.prune.vanilla import prune_vanilla_elementwise, prune_vanilla_kernelwise, \
    prune_vanilla_filterwise, VanillaPruner


import torch

from slender.prune.vanilla import prune_vanilla_elementwise
from slender.quantize.linear import quantize_linear_fix_zeros
from slender.quantize.fixed_point import quantize_fixed_point
from slender.quantize.quantizer import Quantizer
from slender.coding.encode import EncodedParam
from slender.coding.codec import Codec

import math
import random
import torch
from sklearn.linear_model import Lasso


num_pruned_tolerate_coeff = 1.1


def channel_selection(sparsity, output_feature, fn_next_output_feature, method='greedy'):
    """
    select channel to prune with a given metric
    :param sparsity: float, pruning sparsity
    :param output_feature: torch.(cuda.)Tensor, output feature map of the layer being pruned
    :param fn_next_output_feature: function, function to calculate the next output feature map
    :param method: str
                    'greedy': select one contributed to the smallest next feature after another
                    'lasso': select pruned channels by lasso regression
                    'random': randomly select
    :return:
        list of int, indices of filters to be pruned
    """
    num_channel = output_feature.size(1)
    num_pruned = int(math.floor(num_channel * sparsity))


def test_encode_param():
    param = torch.rand(256, 128, 3, 3)
    prune_vanilla_elementwise(sparsity=0.7, param=param)
    quantize_linear_fix_zeros(param, k=16)
    huffman = EncodedParam(param=param, method='huffman',
                           encode_indices=True, bit_length_zero_run_length=4)
    stats = huffman.stats
    print(stats)
    assert torch.eq(param, huffman.data).all()
    state_dict = huffman.state_dict()
    huffman = EncodedParam()
    huffman.load_state_dict(state_dict)
    assert torch.eq(param, huffman.data).all()
    vanilla = EncodedParam(param=param, method='vanilla',
                           encode_indices=True, bit_length_zero_run_length=4)
    stats = vanilla.stats
    print(stats)
    assert torch.eq(param, vanilla.data).all()
    quantize_fixed_point(param=param, bit_length=4, bit_length_integer=0)
    fixed_point = EncodedParam(param=param, method='fixed_point',
                               bit_length=4, bit_length_integer=0,
                               encode_indices=True, bit_length_zero_run_length=4)
    stats = fixed_point.stats
    print(stats)
    assert torch.eq(param, fixed_point.data).all()
    
     if method == 'greedy':
        indices_pruned = []
        while len(indices_pruned) < num_pruned:
            min_diff = 1e10
            min_idx = 0
            for idx in range(num_channel):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                output_feature_try = torch.zeros_like(output_feature)
                output_feature_try[:, indices_try, ...] = output_feature[:, indices_try, ...]
                output_feature_try = fn_next_output_feature(output_feature_try)
                output_feature_try_norm = output_feature_try.norm(2)
                if output_feature_try_norm < min_diff:
                    min_diff = output_feature_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)
    elif method == 'lasso':
        next_output_feature = fn_next_output_feature(output_feature)
        num_el = next_output_feature.numel()
        next_output_feature = next_output_feature.data.view(num_el).cpu()
        next_output_feature_divided = []
        for idx in range(num_channel):
            output_feature_try = torch.zeros_like(output_feature)
            output_feature_try[:, idx, ...] = output_feature[:, idx, ...]
            output_feature_try = fn_next_output_feature(output_feature_try)
            next_output_feature_divided.append(output_feature_try.data.view(num_el, 1))
        next_output_feature_divided = torch.cat(next_output_feature_divided, dim=1).cpu()

        alpha = 5e-5
        solver = Lasso(alpha=alpha, warm_start=True, selection='random')


    
    def test_codec():
    quantize_rule = [
        ('0.weight', 'k-means', 4, 'k-means++'),
        ('1.weight', 'fixed_point', 6, 1),
    ]
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    mask_dict = {}
    for n, p in model.named_parameters():
        mask_dict[n] = prune_vanilla_elementwise(sparsity=0.6, param=p.data)
    quantizer = Quantizer(rule=quantize_rule, fix_zeros=True)
    quantizer.quantize(model, update_labels=False, verbose=True)
    rule = [
        ('0.weight', 'huffman', 0, 0, 4),
        ('1.weight', 'fixed_point', 6, 1, 4)
    ]
    codec = Codec(rule=rule)
    encoded_module = codec.encode(model)
    print(codec.stats)
    state_dict = encoded_module.state_dict()
    model_2 = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                  torch.nn.Conv2d(128, 512, 1, bias=False))
    model_2 = Codec.decode(model_2, state_dict)
    for p1, p2 in zip(model.parameters(), model_2.parameters()):
        if p1.dim() > 1:
            assert torch.eq(p1, p2).all()

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
    
def test_vanilla_pruner():
    rule = [
        ('0.weight', 'element', [0.3, 0.5]),
        ('1.weight', 'element', [0.4, 0.6])
    ]
    rule_dict = {
        '0.weight': [0.3, 0.5],
        '1.weight': [0.4, 0.6]
    }
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    pruner = VanillaPruner(rule=rule)
    pruner.prune(model=model, stage=0, verbose=True)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][0]))
            assert param.data.masked_select(mask).eq(0).all()
    state_dict = pruner.state_dict()
    pruner = VanillaPruner().load_state_dict(state_dict)
    model = torch.nn.Sequential(torch.nn.Conv2d(256, 128, 3, bias=True),
                                torch.nn.Conv2d(128, 512, 1, bias=False))
    
    
        pruner.prune(model=model, stage=0)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][0]))
            assert param.data.masked_select(mask).eq(0).all()
    pruner.prune(model=model, stage=1, update_masks=True, verbose=True)
    for n, param in model.named_parameters():
        if param.dim() > 1:
            mask = pruner.masks[n]
            assert mask.sum() == int(math.ceil(param.numel() * rule_dict[n][1]))
            assert param.data.masked_select(mask).eq(0).all()
