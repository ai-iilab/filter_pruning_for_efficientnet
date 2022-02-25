#Back-bone, regressor, classifier, bifpn module compression
import torch
import numpy
from numpy import linalg
import argparse 
import tensorly  
import yaml

def pruning_efficient_net(model):
    for param in model.parameters():
        param.requires_grad = True
    original_parameters = sum(p.numel()  for p in model.parameters())
    
    blocks = range(0,7)

    for i in blocks:
        if i == 0:
            ex_conv = model._modules['backbone_net']._modules['model']._modules['_conv_stem']._modules['conv'] 
            bn0 = model._modules['backbone_net']._modules['model']._modules['_bn0']
        else: 
            ex_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_expand_conv']._modules['conv']
            bn0 = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_bn0']
        dw_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_depthwise_conv']._modules['conv'] 
        bn1 = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_bn1']
        rd_conv = model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_project_conv']._modules['conv']

        prune_list = get_prune_list(dw_conv,rd_conv, args.comp_ratio)

        print("------------------------------------")
        print(ex_conv.weight.data.size())
        print(dw_conv.weight.data.size())
        #print(se_rd_conv.weight.data.size())
        #print(se_ex_conv.weight.data.size())
        print(rd_conv.weight.data.size())
        print("-------------------------------------") 
        pruned_ex_conv = ex_conv_pruning(ex_conv, prune_list)
        pruned_bn0 = bn_pruning(bn0, prune_list) 
        pruned_dw_conv = dw_conv_pruning(dw_conv, prune_list)
        pruned_bn1 = bn_pruning(bn1, prune_list) 
        #pruned_se_rd_conv = rd_conv_pruning(se_rd_conv, prune_list)
        #pruned_se_ex_conv = ex_conv_pruning(se_ex_conv, prune_list) 
        pruned_rd_conv = rd_conv_pruning(rd_conv, prune_list)
        print(pruned_ex_conv.weight.data.size()) 
        print(pruned_dw_conv.weight.data.size())
        #print(pruned_se_rd_conv.weight.data.size())
        #print(pruned_se_ex_conv.weight.data.size())
        print(pruned_rd_conv.weight.data.size())
        print("-------------------------------------")


        if i == 0:
            model._modules['backbone_net']._modules['model']._modules['_conv_stem']._modules['conv'] =pruned_ex_conv 
            model._modules['backbone_net']._modules['model']._modules['_bn0'] = pruned_bn0 
        else: 
            model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_expand_conv']._modules['conv'] = pruned_ex_conv
            model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_bn0']=pruned_bn0
        model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_depthwise_conv']._modules['conv'] = pruned_dw_conv
        model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_bn1']=pruned_bn1
        #model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_se_reduce']._modules['conv'] = pruned_se_rd_conv
        #model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_se_expand']._modules['conv'] = pruned_se_ex_conv
        model._modules['backbone_net']._modules['model']._modules['_blocks']._modules[str(i)]._modules['_project_conv']._modules['conv'] = pruned_rd_conv

    pruned_parameters = sum(p.numel()  for p in model.parameters())
    compression_rate = pruned_parameters
    pruned_model = model 
    return compression_rate, original_parameters, pruned_parameters, pruned_model


def pruning_bifpn(model):
    for i in range(0,2):
        print(i)
        conv6_up = model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['depthwise_conv']._modules['conv']
        conv6_pt = model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['pointwise_conv']._modules['conv']
        conv6_bn = model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['bn']
        conv5_up = model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['depthwise_conv']._modules['conv']
        conv5_pt = model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['pointwise_conv']._modules['conv']
        conv5_bn = model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['bn']
        conv4_up = model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['depthwise_conv']._modules['conv'] 
        conv4_pt = model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['pointwise_conv']._modules['conv']
        conv4_bn = model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['bn']
        conv3_up = model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['depthwise_conv']._modules['conv']
        conv3_pt = model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['pointwise_conv']._modules['conv']
        conv3_bn = model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['bn']
        conv4_dw = model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['depthwise_conv']._modules['conv']
        conv4_dw_pt = model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['pointwise_conv']._modules['conv']
        conv4_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['bn']
        conv5_dw = model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['depthwise_conv']._modules['conv']
        conv5_dw_pt = model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['pointwise_conv']._modules['conv']
        conv5_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['bn']
        conv6_dw = model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['depthwise_conv']._modules['conv']
        conv6_dw_pt = model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['pointwise_conv']._modules['conv']
        conv6_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['bn']
        conv7_dw = model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['depthwise_conv']._modules['conv']
        conv7_dw_pt = model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['pointwise_conv']._modules['conv']
        conv7_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['bn']
        
        if i == 0:
            p5_dw = model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel']._modules['0']._modules['conv']
            p5_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel']._modules['1']
            p4_dw = model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel']._modules['0']._modules['conv']
            p4_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel']._modules['1']
            p3_dw = model._modules['bifpn']._modules[str(i)]._modules['p3_down_channel']._modules['0']._modules['conv']
            p3_dw_bn = model._modules['bifpn']._modules[str(i)]._modules['p3_down_channel']._modules['1']
            p56 = model._modules['bifpn']._modules[str(i)]._modules['p5_to_p6']._modules['0']._modules['conv']
            p56_bn = model._modules['bifpn']._modules[str(i)]._modules['p5_to_p6']._modules['1'] 
            #p67 = model._modules['bifpn']._modules[str(i)]._modules['p6_to_p7']._modules['0']._modules['conv']
            #p67_bn = model._modules['bifpn']._modules[str(i)]._modules['p5_to_p7']._modules['1']
            p4_dw_2 = model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel_2']._modules['0']._modules['conv'] 
            p4_dw_2_bn = model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel_2']._modules['1']
            p5_dw_2 = model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel_2']._modules['0']._modules['conv']
            p5_dw_2_bn = model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel_2']._modules['1']

        conv6_prune_list = get_prune_list(conv6_up, conv6_pt, args.comp_ratio) 
        conv5_prune_list = get_prune_list(conv5_up, conv5_pt, args.comp_ratio)
        conv4_prune_list = get_prune_list(conv4_up, conv4_pt, args.comp_ratio)
        conv3_prune_list = get_prune_list(conv5_up, conv3_pt, args.comp_ratio) 
        conv4_dw_prune_list = get_prune_list(conv4_dw, conv4_dw_pt, args.comp_ratio)
        conv5_dw_prune_list = get_prune_list(conv5_dw, conv5_dw_pt, args.comp_ratio)
        conv6_dw_prune_list = get_prune_list(conv6_dw, conv6_dw_pt, args.comp_ratio)
        conv7_dw_prune_list = get_prune_list(conv7_dw, conv7_dw_pt, args.comp_ratio) 
        
        pruned_conv6_up = dw_conv_pruning(conv6_up, conv6_prune_list) 
        pruned_conv6_pt = rd_conv_pruning(conv6_pt, conv6_prune_list)
        pruned_conv6_pt = ex_conv_pruning(pruned_conv6_pt, conv6_prune_list)
        pruned_conv6_bn = bn_pruning(conv6_bn, conv6_prune_list)

        pruned_conv5_up = dw_conv_pruning(conv5_up, conv5_prune_list)
        pruned_conv5_pt = rd_conv_pruning(conv5_pt, conv5_prune_list)
        pruned_conv5_pt = ex_conv_pruning(pruned_conv5_pt, conv5_prune_list)
        pruned_conv5_bn = bn_pruning(conv5_bn, conv5_prune_list)

        pruned_conv4_up = dw_conv_pruning(conv4_up, conv4_prune_list)
        pruned_conv4_pt = rd_conv_pruning(conv4_pt, conv4_prune_list)
        pruned_conv4_pt = ex_conv_pruning(pruned_conv4_pt, conv4_prune_list)
        pruned_conv4_bn = bn_pruning(conv4_bn, conv4_prune_list)

        pruned_conv3_up = dw_conv_pruning(conv3_up, conv3_prune_list)
        pruned_conv3_pt = rd_conv_pruning(conv3_pt, conv3_prune_list)
        pruned_conv3_pt = ex_conv_pruning(pruned_conv3_pt, conv3_prune_list)
        pruned_conv3_bn = bn_pruning(conv3_bn, conv3_prune_list)

        pruned_conv4_dw = dw_conv_pruning(conv4_dw, conv4_dw_prune_list)
        pruned_conv4_dw_pt = rd_conv_pruning(conv4_dw_pt, conv4_dw_prune_list)
        pruned_conv4_dw_pt = ex_conv_pruning(pruned_conv4_dw_pt, conv4_dw_prune_list)
        pruned_conv4_dw_bn = bn_pruning(conv4_dw_bn, conv4_dw_prune_list)

        pruned_conv5_dw = dw_conv_pruning(conv5_dw, conv5_dw_prune_list)
        pruned_conv5_dw_pt = rd_conv_pruning(conv5_dw_pt, conv5_dw_prune_list)
        pruned_conv5_dw_pt = ex_conv_pruning(pruned_conv5_dw_pt, conv5_dw_prune_list)
        pruned_conv5_dw_bn = bn_pruning(conv5_dw_bn, conv5_dw_prune_list)

        pruned_conv6_dw = dw_conv_pruning(conv6_dw, conv6_dw_prune_list)
        pruned_conv6_dw_pt = rd_conv_pruning(conv6_dw_pt, conv6_dw_prune_list)
        pruned_conv6_dw_pt = ex_conv_pruning(pruned_conv6_dw_pt, conv6_dw_prune_list)
        pruned_conv6_dw_bn = bn_pruning(conv6_dw_bn, conv6_dw_prune_list)

        pruned_conv7_dw = dw_conv_pruning(conv7_dw, conv7_dw_prune_list)
        pruned_conv7_dw_pt = rd_conv_pruning(conv7_dw_pt, conv7_dw_prune_list)
        pruned_conv7_dw_pt = ex_conv_pruning(pruned_conv7_dw_pt, conv7_dw_prune_list)
        pruned_conv7_dw_bn = bn_pruning(conv7_dw_bn, conv7_dw_prune_list)

        if i == 0:
            pruned_p5_dw = ex_conv_pruning(p5_dw, conv5_dw_prune_list)
            pruned_p5_dw_bn = bn_pruning(p5_dw_bn, conv5_dw_prune_list) 
            pruned_p4_dw = ex_conv_pruning(p4_dw, conv4_dw_prune_list) 
            pruned_p4_dw_bn = bn_pruning(p4_dw_bn, conv4_dw_prune_list) 
            pruned_p3_dw = ex_conv_pruning(p3_dw, conv3_prune_list) 
            pruned_p3_dw_bn = bn_pruning(p3_dw_bn, conv3_prune_list)
            pruned_p56 = ex_conv_pruning(p56, conv5_dw_prune_list) 
            pruned_p56_bn = bn_pruning(p56_bn, conv5_dw_prune_list) 
            #pruned_p67 = rd_conv_pruning(p67, conv6_dw_prune_list)
            #pruned_p67_bn = bn_pruning(p67_bn, conv6_dw_prune_list)
            pruned_p4_dw_2 = ex_conv_pruning(p4_dw_2, conv4_dw_prune_list) 
            pruned_p4_dw_2_bn  = bn_pruning(p4_dw_2_bn, conv4_dw_prune_list)
            pruned_p5_dw_2 = ex_conv_pruning(p5_dw_2, conv5_dw_prune_list) 
            pruned_p5_dw_2_bn = bn_pruning(p5_dw_2_bn, conv5_dw_prune_list)

        model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['depthwise_conv']._modules['conv'] = pruned_conv6_up
        model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['pointwise_conv']._modules['conv'] = pruned_conv6_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv6_up']._modules['bn'] = pruned_conv6_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['depthwise_conv']._modules['conv'] = pruned_conv5_up
        model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['pointwise_conv']._modules['conv'] = pruned_conv5_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv5_up']._modules['bn'] = pruned_conv5_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['depthwise_conv']._modules['conv'] = pruned_conv4_up
        model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['pointwise_conv']._modules['conv'] = pruned_conv4_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv4_up']._modules['bn'] = pruned_conv4_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['depthwise_conv']._modules['conv'] = pruned_conv3_up
        model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['pointwise_conv']._modules['conv'] = pruned_conv3_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv3_up']._modules['bn'] = pruned_conv3_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['depthwise_conv']._modules['conv'] = pruned_conv4_dw
        model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['pointwise_conv']._modules['conv'] = pruned_conv4_dw_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv4_down']._modules['bn'] = pruned_conv4_dw_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['depthwise_conv']._modules['conv'] = pruned_conv5_dw
        model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['pointwise_conv']._modules['conv'] = pruned_conv5_dw_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv5_down']._modules['bn'] = pruned_conv5_dw_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['depthwise_conv']._modules['conv'] = pruned_conv6_dw
        model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['pointwise_conv']._modules['conv'] = pruned_conv6_dw_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv6_down']._modules['bn'] = pruned_conv6_dw_bn
        model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['depthwise_conv']._modules['conv'] = pruned_conv7_dw
        model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['pointwise_conv']._modules['conv'] = pruned_conv7_dw_pt
        model._modules['bifpn']._modules[str(i)]._modules['conv7_down']._modules['bn'] = pruned_conv7_dw_bn
        if i == 0:
            model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel']._modules['0']._modules['conv'] = pruned_p5_dw
            model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel']._modules['1'] = pruned_p5_dw_bn
            model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel']._modules['0']._modules['conv'] = pruned_p4_dw
            model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel']._modules['1'] = pruned_p4_dw_bn
            model._modules['bifpn']._modules[str(i)]._modules['p3_down_channel']._modules['0']._modules['conv'] = pruned_p3_dw
            model._modules['bifpn']._modules[str(i)]._modules['p3_down_channel']._modules['1'] = pruned_p3_dw_bn
            model._modules['bifpn']._modules[str(i)]._modules['p5_to_p6']._modules['0']._modules['conv'] = pruned_p56
            model._modules['bifpn']._modules[str(i)]._modules['p5_to_p6']._modules['1'] = pruned_p56_bn
            #model._modules['bifpn']._modules[str(i)]._modules['p6_to_p7']._modules['0']._modules['conv'] = pruned_p67
            #model._modules['bifpn']._modules[str(i)]._modules['p6_to_p7']._modules['1'] = pruned_p67_bn
            model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel_2']._modules['0']._modules['conv'] = pruned_p4_dw_2
            model._modules['bifpn']._modules[str(i)]._modules['p4_down_channel_2']._modules['1'] = pruned_p4_dw_2_bn 
            model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel_2']._modules['0']._modules['conv'] = pruned_p5_dw_2 
            model._modules['bifpn']._modules[str(i)]._modules['p5_down_channel_2']._modules['1'] = pruned_p5_dw_2_bn

    return model

def pruning_classifier(model):
    conv0 = model._modules['classifier']._modules['conv_list']._modules['0']._modules['depthwise_conv']._modules['conv']
    conv0_pt = model._modules['classifier']._modules['conv_list']._modules['0']._modules['pointwise_conv']._modules['conv']
    conv1 = model._modules['classifier']._modules['conv_list']._modules['1']._modules['depthwise_conv']._modules['conv']
    conv1_pt = model._modules['classifier']._modules['conv_list']._modules['1']._modules['pointwise_conv']._modules['conv']
    conv2 = model._modules['classifier']._modules['conv_list']._modules['2']._modules['depthwise_conv']._modules['conv']
    conv2_pt = model._modules['classifier']._modules['conv_list']._modules['2']._modules['pointwise_conv']._modules['conv']
    bn00 = model._modules['classifier']._modules['bn_list']._modules['0']._modules['0']
    bn01 = model._modules['classifier']._modules['bn_list']._modules['0']._modules['1']
    bn02 = model._modules['classifier']._modules['bn_list']._modules['0']._modules['2']
    bn10 = model._modules['classifier']._modules['bn_list']._modules['1']._modules['0']
    bn11 = model._modules['classifier']._modules['bn_list']._modules['1']._modules['1']
    bn12 = model._modules['classifier']._modules['bn_list']._modules['1']._modules['2']
    bn20 = model._modules['classifier']._modules['bn_list']._modules['2']._modules['0']
    bn21 = model._modules['classifier']._modules['bn_list']._modules['2']._modules['1']
    bn22 = model._modules['classifier']._modules['bn_list']._modules['2']._modules['2']
    bn30 = model._modules['classifier']._modules['bn_list']._modules['3']._modules['0']
    bn31 = model._modules['classifier']._modules['bn_list']._modules['3']._modules['1']
    bn32 = model._modules['classifier']._modules['bn_list']._modules['3']._modules['2']
    bn40 = model._modules['classifier']._modules['bn_list']._modules['4']._modules['0']
    bn41 = model._modules['classifier']._modules['bn_list']._modules['4']._modules['1']
    bn42 = model._modules['classifier']._modules['bn_list']._modules['4']._modules['2']
    head_conv = model._modules['classifier']._modules['header']._modules['depthwise_conv']._modules['conv']
    head_conv_pt = model._modules['classifier']._modules['header']._modules['pointwise_conv']._modules['conv']

    cls_prune_list = get_prune_list(conv0, conv0_pt, args.comp_ratio)

    pruned_conv0 = dw_conv_pruning(conv0, cls_prune_list)
    pruned_conv0_pt = rd_conv_pruning(conv0_pt, cls_prune_list)
    pruned_conv0_pt = ex_conv_pruning(pruned_conv0_pt, cls_prune_list)
    pruned_conv1 = dw_conv_pruning(conv1, cls_prune_list)
    pruned_conv1_pt = rd_conv_pruning(conv1_pt, cls_prune_list)
    pruned_conv1_pt = ex_conv_pruning(pruned_conv1_pt, cls_prune_list)
    pruned_conv2 = dw_conv_pruning(conv2, cls_prune_list)
    pruned_conv2_pt = rd_conv_pruning(conv2_pt, cls_prune_list)
    pruned_conv2_pt = ex_conv_pruning(pruned_conv2_pt, cls_prune_list)
    pruned_bn00 = bn_pruning(bn00, cls_prune_list)
    pruned_bn01 = bn_pruning(bn01, cls_prune_list)
    pruned_bn02 = bn_pruning(bn02, cls_prune_list)
    pruned_bn10 = bn_pruning(bn10, cls_prune_list)
    pruned_bn11 = bn_pruning(bn11, cls_prune_list)
    pruned_bn12 = bn_pruning(bn12, cls_prune_list)
    pruned_bn20 = bn_pruning(bn20, cls_prune_list)
    pruned_bn21 = bn_pruning(bn21, cls_prune_list)
    pruned_bn22 = bn_pruning(bn22, cls_prune_list)
    pruned_bn30 = bn_pruning(bn30, cls_prune_list)
    pruned_bn31 = bn_pruning(bn31, cls_prune_list)
    pruned_bn32 = bn_pruning(bn32, cls_prune_list)
    pruned_bn40 = bn_pruning(bn40, cls_prune_list)
    pruned_bn41 = bn_pruning(bn41, cls_prune_list)
    pruned_bn42 = bn_pruning(bn42, cls_prune_list)
    pruned_head_conv = dw_conv_pruning(head_conv, cls_prune_list)
    pruned_head_conv_pt = rd_conv_pruning(head_conv_pt, cls_prune_list)

    model._modules['classifier']._modules['conv_list']._modules['0']._modules['depthwise_conv']._modules['conv'] = pruned_conv0
    model._modules['classifier']._modules['conv_list']._modules['0']._modules['pointwise_conv']._modules['conv'] = pruned_conv0_pt
    model._modules['classifier']._modules['conv_list']._modules['1']._modules['depthwise_conv']._modules['conv'] = pruned_conv1
    model._modules['classifier']._modules['conv_list']._modules['1']._modules['pointwise_conv']._modules['conv'] = pruned_conv1_pt
    model._modules['classifier']._modules['conv_list']._modules['2']._modules['depthwise_conv']._modules['conv'] = pruned_conv2
    model._modules['classifier']._modules['conv_list']._modules['2']._modules['pointwise_conv']._modules['conv'] = pruned_conv2_pt
    model._modules['classifier']._modules['bn_list']._modules['0']._modules['0'] = pruned_bn00
    model._modules['classifier']._modules['bn_list']._modules['0']._modules['1'] = pruned_bn01
    model._modules['classifier']._modules['bn_list']._modules['0']._modules['2'] = pruned_bn02
    model._modules['classifier']._modules['bn_list']._modules['1']._modules['0'] = pruned_bn10
    model._modules['classifier']._modules['bn_list']._modules['1']._modules['1'] = pruned_bn11
    model._modules['classifier']._modules['bn_list']._modules['1']._modules['2'] = pruned_bn12
    model._modules['classifier']._modules['bn_list']._modules['2']._modules['0'] = pruned_bn20
    model._modules['classifier']._modules['bn_list']._modules['2']._modules['1'] = pruned_bn21
    model._modules['classifier']._modules['bn_list']._modules['2']._modules['2'] = pruned_bn22
    model._modules['classifier']._modules['bn_list']._modules['3']._modules['0'] = pruned_bn30
    model._modules['classifier']._modules['bn_list']._modules['3']._modules['1'] = pruned_bn31
    model._modules['classifier']._modules['bn_list']._modules['3']._modules['2'] = pruned_bn32
    model._modules['classifier']._modules['bn_list']._modules['4']._modules['0'] = pruned_bn40
    model._modules['classifier']._modules['bn_list']._modules['4']._modules['1'] = pruned_bn41
    model._modules['classifier']._modules['bn_list']._modules['4']._modules['2'] = pruned_bn42
    model._modules['classifier']._modules['header']._modules['depthwise_conv']._modules['conv'] = pruned_head_conv
    model._modules['classifier']._modules['header']._modules['pointwise_conv']._modules['conv'] = pruned_head_conv_pt

    return model


def pruning_regressor(model):
    conv0 = model._modules['regressor']._modules['conv_list']._modules['0']._modules['depthwise_conv']._modules['conv']
    conv0_pt = model._modules['regressor']._modules['conv_list']._modules['0']._modules['pointwise_conv']._modules['conv']
    conv1 = model._modules['regressor']._modules['conv_list']._modules['1']._modules['depthwise_conv']._modules['conv']
    conv1_pt = model._modules['regressor']._modules['conv_list']._modules['1']._modules['pointwise_conv']._modules['conv']
    conv2 = model._modules['regressor']._modules['conv_list']._modules['2']._modules['depthwise_conv']._modules['conv']
    conv2_pt = model._modules['regressor']._modules['conv_list']._modules['2']._modules['pointwise_conv']._modules['conv']
    bn00 = model._modules['regressor']._modules['bn_list']._modules['0']._modules['0']
    bn01 = model._modules['regressor']._modules['bn_list']._modules['0']._modules['1']
    bn02 = model._modules['regressor']._modules['bn_list']._modules['0']._modules['2']
    bn10 = model._modules['regressor']._modules['bn_list']._modules['1']._modules['0']
    bn11 = model._modules['regressor']._modules['bn_list']._modules['1']._modules['1']
    bn12 = model._modules['regressor']._modules['bn_list']._modules['1']._modules['2']
    bn20 = model._modules['regressor']._modules['bn_list']._modules['2']._modules['0']
    bn21 = model._modules['regressor']._modules['bn_list']._modules['2']._modules['1']
    bn22 = model._modules['regressor']._modules['bn_list']._modules['2']._modules['2']
    bn30 = model._modules['regressor']._modules['bn_list']._modules['3']._modules['0']
    bn31 = model._modules['regressor']._modules['bn_list']._modules['3']._modules['1']
    bn32 = model._modules['regressor']._modules['bn_list']._modules['3']._modules['2']
    bn40 = model._modules['regressor']._modules['bn_list']._modules['4']._modules['0']
    bn41 = model._modules['regressor']._modules['bn_list']._modules['4']._modules['1']
    bn42 = model._modules['regressor']._modules['bn_list']._modules['4']._modules['2']
    head_conv = model._modules['regressor']._modules['header']._modules['depthwise_conv']._modules['conv']
    head_conv_pt = model._modules['regressor']._modules['header']._modules['pointwise_conv']._modules['conv']

    cls_prune_list = get_prune_list(conv0, conv0_pt, args.comp_ratio)

    pruned_conv0 = dw_conv_pruning(conv0, cls_prune_list)
    pruned_conv0_pt = rd_conv_pruning(conv0_pt, cls_prune_list)
    pruned_conv0_pt = ex_conv_pruning(pruned_conv0_pt, cls_prune_list)
    pruned_conv1 = dw_conv_pruning(conv1, cls_prune_list)
    pruned_conv1_pt = rd_conv_pruning(conv1_pt, cls_prune_list)
    pruned_conv1_pt = ex_conv_pruning(pruned_conv1_pt, cls_prune_list)
    pruned_conv2 = dw_conv_pruning(conv2, cls_prune_list)
    pruned_conv2_pt = rd_conv_pruning(conv2_pt, cls_prune_list)
    pruned_conv2_pt = ex_conv_pruning(pruned_conv2_pt, cls_prune_list)
    pruned_bn00 = bn_pruning(bn00, cls_prune_list)
    pruned_bn01 = bn_pruning(bn01, cls_prune_list)
    pruned_bn02 = bn_pruning(bn02, cls_prune_list)
    pruned_bn10 = bn_pruning(bn10, cls_prune_list)
    pruned_bn11 = bn_pruning(bn11, cls_prune_list)
    pruned_bn12 = bn_pruning(bn12, cls_prune_list)
    pruned_bn20 = bn_pruning(bn20, cls_prune_list)
    pruned_bn21 = bn_pruning(bn21, cls_prune_list)
    pruned_bn22 = bn_pruning(bn22, cls_prune_list)
    pruned_bn30 = bn_pruning(bn30, cls_prune_list)
    pruned_bn31 = bn_pruning(bn31, cls_prune_list)
    pruned_bn32 = bn_pruning(bn32, cls_prune_list)
    pruned_bn40 = bn_pruning(bn40, cls_prune_list)
    pruned_bn41 = bn_pruning(bn41, cls_prune_list)
    pruned_bn42 = bn_pruning(bn42, cls_prune_list)
    pruned_head_conv = dw_conv_pruning(head_conv, cls_prune_list)
    pruned_head_conv_pt = rd_conv_pruning(head_conv_pt, cls_prune_list)

    model._modules['regressor']._modules['conv_list']._modules['0']._modules['depthwise_conv']._modules['conv'] = pruned_conv0
    model._modules['regressor']._modules['conv_list']._modules['0']._modules['pointwise_conv']._modules['conv'] = pruned_conv0_pt
    model._modules['regressor']._modules['conv_list']._modules['1']._modules['depthwise_conv']._modules['conv'] = pruned_conv1
    model._modules['regressor']._modules['conv_list']._modules['1']._modules['pointwise_conv']._modules['conv'] = pruned_conv1_pt
    model._modules['regressor']._modules['conv_list']._modules['2']._modules['depthwise_conv']._modules['conv'] = pruned_conv2
    model._modules['regressor']._modules['conv_list']._modules['2']._modules['pointwise_conv']._modules['conv'] = pruned_conv2_pt
    model._modules['regressor']._modules['bn_list']._modules['0']._modules['0'] = pruned_bn00
    model._modules['regressor']._modules['bn_list']._modules['0']._modules['1'] = pruned_bn01
    model._modules['regressor']._modules['bn_list']._modules['0']._modules['2'] = pruned_bn02
    model._modules['regressor']._modules['bn_list']._modules['1']._modules['0'] = pruned_bn10
    model._modules['regressor']._modules['bn_list']._modules['1']._modules['1'] = pruned_bn11
    model._modules['regressor']._modules['bn_list']._modules['1']._modules['2'] = pruned_bn12
    model._modules['regressor']._modules['bn_list']._modules['2']._modules['0'] = pruned_bn20
    model._modules['regressor']._modules['bn_list']._modules['2']._modules['1'] = pruned_bn21
    model._modules['regressor']._modules['bn_list']._modules['2']._modules['2'] = pruned_bn22
    model._modules['regressor']._modules['bn_list']._modules['3']._modules['0'] = pruned_bn30
    model._modules['regressor']._modules['bn_list']._modules['3']._modules['1'] = pruned_bn31
    model._modules['regressor']._modules['bn_list']._modules['3']._modules['2'] = pruned_bn32
    model._modules['regressor']._modules['bn_list']._modules['4']._modules['0'] = pruned_bn40
    model._modules['regressor']._modules['bn_list']._modules['4']._modules['1'] = pruned_bn41
    model._modules['regressor']._modules['bn_list']._modules['4']._modules['2'] = pruned_bn42
    model._modules['regressor']._modules['header']._modules['depthwise_conv']._modules['conv'] = pruned_head_conv
    model._modules['regressor']._modules['header']._modules['pointwise_conv']._modules['conv'] = pruned_head_conv_pt

    return model
