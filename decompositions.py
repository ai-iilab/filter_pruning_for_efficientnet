import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn


def VBsigma2(sigma2,L,M,cacb,s,residual):
    H = len(s)

    thresh_term = (L+M + sigma2/cacb**2)/2
    threshold = np.sqrt( sigma2 * (thresh_term + np.sqrt(thresh_term**2 - L*M) ))
    pos = np.sum(s>threshold)

    d = np.multiply(s[:pos],
                    1 - np.multiply(sigma2/(2*s[:pos]**2),
                                    L+M+np.sqrt( (M-L)**2 + 4*s[:pos]**2/cacb**2 )))

    zeta = sigma2/(2*L*M) * (L+M+sigma2/cacb**2 - np.sqrt((L+M+sigma2/cacb**2)**2 - 4*L*M))
    post_ma = np.zeros(H)
    post_mb = np.zeros(H)
    post_sa2 = cacb * (1-L*zeta/sigma2) * np.ones(H)
    post_sb2 = cacb * (1-M*zeta/sigma2) * np.ones(H)

    delta = cacb/sigma2 * (s[:pos]-d- L*sigma2/s[:pos])
    post_ma[:pos] = np.sqrt(np.multiply(d, delta))
    post_mb[:pos] = np.sqrt(np.divide(d, delta))
    post_sa2[:pos] = np.divide(sigma2*delta, s[:pos])
    post_sb2[:pos] = np.divide(sigma2, np.multiply(delta, s[:pos]))

    F = 0.5*(L*M*np.log(2*np.pi*sigma2) + (residual+np.sum(s**2))/sigma2 - (L+M)*H
               + np.sum(M*np.log(cacb/post_sa2) + L*np.log(cacb/post_sb2)
                        + (post_ma**2 + M*post_sa2)/cacb + (post_mb**2 + L*post_sb2)/cacb
                        + (-2 * np.multiply(np.multiply(post_ma, post_mb), s)
                           + np.multiply(post_ma**2 + M*post_sa2,post_mb**2 + L*post_sb2))/sigma2))
    return F



def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly. 
    last, first, vertical, horizontal = \
        parafac(layer.weight.data, rank=rank, init='svd')

    pointwise_s_to_r_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(in_channels=vertical.shape[1], 
            out_channels=vertical.shape[1], kernel_size=(vertical.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), dilation=layer.dilation,
            groups=vertical.shape[1], bias=False)

    depthwise_horizontal_layer = \
        torch.nn.Conv2d(in_channels=horizontal.shape[1], \
            out_channels=horizontal.shape[1], 
            kernel_size=(1, horizontal.shape[0]), stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
            out_channels=last.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = \
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, \
                    depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    return nn.Sequential(*new_layers)

def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    #import pdb
    #pdb.set_trace(a)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    if ranks[0] == 0:
       ranks[0] =1
    if ranks[1] ==0:
       ranks[1] =1
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """
    
    ranks = estimate_ranks(layer)
             
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data, \
            modes=[0, 1], ranks=ranks, init='svd')
    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)
    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)
    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data
    
    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)
