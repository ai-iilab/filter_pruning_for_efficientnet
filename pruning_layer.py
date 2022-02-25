import torch
import numpy 
from numpy import linalg
import tensorly

from calculate_filter_importance import get_prune_list

# dimension expansion layer pruning (output channel pruning)
def ex_conv_pruning(ex_conv, prune_list):
    pruned_ex_conv = torch.nn.Conv2d(in_channels = ex_conv.in_channels, out_channels = ex_conv.out_channels - len(prune_list), kernel_size=ex_conv.kernel_size, stride = ex_conv.stride, padding = ex_conv.padding, dilation = ex_conv.dilation, bias = (ex_conv.bias is not None))  
    
    pruned_ex_conv_weight = pruned_ex_conv.weight.data.cpu().numpy() 
    ex_conv_weight = ex_conv.weight.data.cpu().numpy() 
     
    pruned_ex_conv_weight = numpy.delete(ex_conv_weight, prune_list, 0) 
    
    pruned_ex_conv.weight.data = torch.from_numpy(pruned_ex_conv_weight)
    pruned_ex_conv.weight.data = pruned_ex_conv.weight.data.cuda()
    
    return pruned_ex_conv
  
# depth-wise convolutional layer pruning (input and output channels pruning)
def dw_conv_pruning(dw_conv, prune_list):
    pruned_dw_conv = torch.nn.Conv2d(in_channels = dw_conv.in_channels - len(prune_list), out_channels = dw_conv.out_channels - len(prune_list), kernel_size=dw_conv.kernel_size, stride=dw_conv.stride, padding = dw_conv.padding, dilation = dw_conv.dilation, groups = dw_conv.groups - len(prune_list),  bias = (dw_conv.bias is not None)) 

    pruned_dw_conv_weight = pruned_dw_conv.weight.data.cpu().numpy()
    dw_conv_weight = dw_conv.weight.data.cpu().numpy()

    pruned_dw_conv_weight = numpy.delete(dw_conv_weight, prune_list, 0)

    pruned_dw_conv.weight.data = torch.from_numpy(pruned_dw_conv_weight)
    pruned_dw_conv.weight.data = pruned_dw_conv.weight.data.cuda() 

    return pruned_dw_conv

# dimension reduction convolutional layer pruning (input channel pruning)
def rd_conv_pruning(rd_conv, prune_list):
    pruned_rd_conv = torch.nn.Conv2d(in_channels = rd_conv.in_channels - len(prune_list), out_channels = rd_conv.out_channels, kernel_size = rd_conv.kernel_size, stride=rd_conv.stride, padding=rd_conv.padding, dilation=rd_conv.dilation, bias = (rd_conv.bias  is not None))

    pruned_rd_conv_weight = pruned_rd_conv.weight.data.cpu().numpy()
    rd_conv_weight = rd_conv.weight.data.cpu().numpy()

    pruned_rd_conv_weight = numpy.delete(rd_conv_weight, prune_list, 1)

    pruned_rd_conv.weight.data = torch.from_numpy(pruned_rd_conv_weight)
    pruned_rd_conv.weight.data = pruned_rd_conv.weight.data.cuda()
    return pruned_rd_conv

# batch-normalization layer pruning
def bn_pruning(bn, prune_list):
    num_features = bn.num_features
    pruned_bn = torch.nn.BatchNorm2d(num_features = bn.num_features - len(prune_list), eps = bn.eps, momentum = bn.momentum, affine=bn.affine, track_running_stats=bn.track_running_stats)

    bn_weight = bn.weight.data.cpu().numpy() 
    pruned_bn_weight = pruned_bn.weight.data.cpu().numpy()

    bn_bias = bn.bias.data.cpu().numpy()
    pruned_bn_bias = pruned_bn.bias.data.cpu().numpy() 

    pruned_bn_weight = numpy.delete(bn_weight, prune_list) 
    pruned_bn_bias = numpy.delete(bn_bias, prune_list) 

    pruned_bn.weight.data = torch.from_numpy(pruned_bn_weight) 
    pruned_bn.weight.data = pruned_bn.weight.data.cuda() 

    pruned_bn.bias.data = torch.from_numpy(pruned_bn_bias) 
    pruned_bn.bias.data = pruned_bn.bias.data.cuda() 
    return pruned_bn 
