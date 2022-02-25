import torch
import numpy
from numpy import linalg
import argparse 
import tensorly  
import yaml 

parser = argparse.ArgumentParser(description='filter selection for efficient net')
parser.add_argument('--load', '-l', type=str, default='./target_model.pth')
parser.add_argument('--save_name', '-n', type=str, default='./pruned_model.pth        
parser.add_argument('--comp_ratio', '-c', type=float, default = '0.5') 
args = parser.parse_args() 

class Params: 
    def __init__ (self, project_file):
        self.params = yaml.safe_load(open(project_file).read())
    
    def __getattr__ (self, item):
        return self.params.get(item, None)   
  
def load_model(load_name):
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=1, onnx_export=True, ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales)).to("cuda")
    model.backbone_net.model.set_swish(memory_efficient=False)
    model.load_state_dict(torch.load(load_name))
    return model
  
model = torch.load(args.load)
print(model)
original_parameters = sum(p.numel()  for p in model.parameters())
_, _, _, pruned_model = pruning_efficient_net(model)
pruned_model = pruning_bifpn(pruned_model)
pruned_model = pruning_regressor(pruned_model)
pruned_model = pruning_classifier(pruned_model)
pruned_parameters = sum(p.numel()  for p in pruned_model.parameters())
torch.save(pruned_model, args.save_name) 
print("compression rate:" + str(pruned_parameters/original_parameters) + " # of original:" + str(original_parameters) + " # of pruned model:" + str(pruned_parameters))
