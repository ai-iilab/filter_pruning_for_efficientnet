import torch
import torchvision.models as models
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from tensorly.decomposition import parafac_power_iteration
import numpy as np
import argparse
import copy

parser = argparse.ArgumentParser(
    description="CNN acceleration via Low-rank tensor dcomposition and Filter-level pruning"
)
parser.add_argument("-model_name", type=str, default="vgg16")  # default를 res18로 바꿔봄
parser.add_argument("-importance_criteria", type=str, default="l2", help="l1, l2, fpgm, ours")
parser.add_argument("-pruning_ratio", type=float, default=0.7)
parser.add_argument("-soft_pruning", type=bool, default=False)
parser.add_argument("-save_name", type=str, default="./test")
parser.add_argument("-save", default=False)
parser.add_argument("-layer",  action='store_true')
args = parser.parse_args()

def get_conv_list_res1(model):
    list = []
    for _, key1 in enumerate(model._modules.keys()):
        if "_x" in key1:
            for _, key2 in enumerate(model._modules[key1]._modules.keys()):
                for _, key3 in enumerate(model._modules[key1]._modules[key2]._modules.keys()):
                    if not model._modules[key1]._modules[key2]._modules[key3]:
                        continue
                    for _, key4 in enumerate(model._modules[key1]._modules[key2]._modules[key3]._modules.keys()):
                        if isinstance(model._modules[key1]._modules[key2]._modules[key3]._modules[key4],torch.nn.modules.conv.Conv2d) or isinstance(model._modules[key1]._modules[key2]._modules[key3]._modules[key4],torch.nn.BatchNorm2d):
                            list.append([key1, key2, key3, key4])
    return list


#original_model = torch.load("")

target_model = nn.Sequential(*(list(original_model.children())[:4]), original_model._modules['conv5_x']._modules['0'])
   

original_params = sum(p.numel() for p in target_model.parameters())
print('memory: ', original_params)
