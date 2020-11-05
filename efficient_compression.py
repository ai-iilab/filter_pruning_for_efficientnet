import torch
import numpy
from numpy import linalg

#load model 
model = torch.load('trained_model')  
print(sum(p.numel()  for p in model.parameters() if p.requires_grad)) 

def ex_conv_pruning(ex_conv, prune_list):
    pruned_ex_conv = torch.nn.Conv2d(in_channels = ex_conv.in_channels, out_channels = ex_conv.out_channels - len(prune_list), kernel_size=ex_conv.kernel_size, stride = ex_conv.stride, padding = ex_conv.padding, dilation = ex_conv.dilation, bias = (ex_conv.bias is not None))  
    
    pruned_ex_conv_weight = pruned_ex_conv.weight.data.cpu().numpy() 
    ex_conv_weight = ex_conv.weight.data.cpu().numpy() 
     
    pruned_ex_conv_weight = numpy.delete(ex_conv_weight, prune_list, 0) 
    
    pruned_ex_conv.weight.data = torch.from_numpy(pruned_ex_conv_weight)
    pruned_ex_conv.weight.data = pruned_ex_conv.weight.data.cuda()
    
    return pruned_ex_conv

def dw_conv_pruning(dw_conv, prune_list):
    pruned_dw_conv = torch.nn.Conv2d(in_channels = dw_conv.in_channels - len(prune_list), out_channels = dw_conv.out_channels - len(prune_list), kernel_size=dw_conv.kernel_size, stride=dw_conv.stride, padding = dw_conv.padding, dilation = dw_conv.dilation, groups = dw_conv.groups - len(prune_list),  bias = (dw_conv.bias is not None)) 

    pruned_dw_conv_weight = pruned_dw_conv.weight.data.cpu().numpy()
    dw_conv_weight = dw_conv.weight.data.cpu().numpy()

    pruned_dw_conv_weight = numpy.delete(dw_conv_weight, prune_list, 0)

    pruned_dw_conv.weight.data = torch.from_numpy(pruned_dw_conv_weight)
    pruned_dw_conv.weight.data = pruned_dw_conv.weight.data.cuda() 

    return pruned_dw_conv

def rd_conv_pruning(rd_conv, prune_list):
    pruned_rd_conv = torch.nn.Conv2d(in_channels = rd_conv.in_channels - len(prune_list), out_channels = rd_conv.out_channels, kernel_size = rd_conv.kernel_size, stride=rd_conv.stride, padding=rd_conv.padding, dilation=rd_conv.dilation, bias = (rd_conv.bias  is not None))

    pruned_rd_conv_weight = pruned_rd_conv.weight.data.cpu().numpy()
    rd_conv_weight = rd_conv.weight.data.cpu().numpy()

    pruned_rd_conv_weight = numpy.delete(rd_conv_weight, prune_list, 1)

    pruned_rd_conv.weight.data = torch.from_numpy(pruned_rd_conv_weight)
    pruned_rd_conv.weight.data = pruned_rd_conv.weight.data.cuda()
    return pruned_rd_conv


def get_prune_list(conv,cut_off_rate):
    importance_list = numpy.array([])
    weight = conv.weight.data.cpu().numpy() 
    for i in range(conv.out_channels):
        weight_matrix = weight[i,:,:,:].squeeze()
        importance_list=numpy.append(importance_list, numpy.linalg.norm(weight_matrix,'nuc'))
    prune_list = sorted(range(len(importance_list)),key=lambda i: importance_list[i])[:int(conv.out_channels*cut_off_rate)] 
    return prune_list


blocks = range(2,23)

for i in blocks:
    ex_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_expand_conv']._modules['conv']
    dw_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_depthwise_conv']._modules['conv'] 
    se_rd_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_se_reduce']._modules['conv']
    se_ex_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_se_expand']._modules['conv']
    rd_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_project_conv']._modules['conv']

    prune_list = get_prune_list(dw_conv, 0.5)

    print("------------------------------------")
    print(ex_conv.weight.data.size())
    print(dw_conv.weight.data.size())
    print(se_rd_conv.weight.data.size())
    print(se_ex_conv.weight.data.size())
    print(rd_conv.weight.data.size())
    print("-------------------------------------") 
    pruned_ex_conv = ex_conv_pruning(ex_conv, prune_list)
    pruned_dw_conv = dw_conv_pruning(dw_conv, prune_list)
    pruned_rd_conv = rd_conv_pruning(rd_conv, prune_list) 
    print(pruned_ex_conv.weight.data.size()) 
    print(pruned_dw_conv.weight.data.size())
    print(pruned_rd_conv.weight.data.size())
    print("-------------------------------------")

    model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_expand_conv']._modules['conv'] = pruned_ex_conv
    model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_depthwise_conv']._modules['conv'] = pruned_dw_conv
    model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_se_reduce']._modules['conv'] = pruned_rd_conv


torch.save(model, "pruned_model")
print(sum(p.numel()  for p in model.parameters() if p.requires_grad))

