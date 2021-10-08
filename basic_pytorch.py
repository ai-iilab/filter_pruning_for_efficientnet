
import torch

def weight_reconstruction(next_module, next_input_feature, next_output_feature, cpu=True):
    """
    reconstruct the weight of the next layer to the one being pruned
    :param next_module: torch.nn.module, module of the next layer to the one being pruned
    :param next_input_feature: torch.(cuda.)Tensor, new input feature map of the next layer
    :param next_output_feature: torch.(cuda.)Tensor, original output feature map of the next layer
    :param cpu: bool, whether done in cpu
    :return:
        void
    """
    if next_module.bias is not None:
        bias_size = [1] * next_output_feature.dim()
        bias_size[1] = -1
        next_output_feature -= next_module.bias.view(bias_size)
    if cpu:
        next_input_feature = next_input_feature.cpu()
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        unfold = torch.nn.Unfold(kernel_size=next_module.kernel_size,
                                 dilation=next_module.dilation,
                                 padding=next_module.padding,
                                 stride=next_module.stride)
        if not cpu:
            unfold = unfold.cuda()
        unfold.eval()
        next_input_feature = unfold(next_input_feature)
        next_input_feature = next_input_feature.transpose(1, 2)
        num_fields = next_input_feature.size(0) * next_input_feature.size(1)
        next_input_feature = next_input_feature.reshape(num_fields, -1)
        next_output_feature = next_output_feature.view(next_output_feature.size(0), next_output_feature.size(1), -1)
        next_output_feature = next_output_feature.transpose(1, 2).reshape(num_fields, -1)
    if cpu:
        next_output_feature = next_output_feature.cpu()
    param, _ = torch.gels(next_output_feature.data, next_input_feature.data)
    param = param[0:next_input_feature.size(1), :].clone().t().contiguous().view(next_output_feature.size(1), -1)
    if isinstance(next_module, torch.nn.modules.conv._ConvNd):
        param = param.view(next_module.out_channels, next_module.in_channels, *next_module.kernel_size)
    del next_module.weight
    next_module.weight = torch.nn.Parameter(param

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
    rule = [
        ('0.weight', 'element', [0.3, 0.5]),
        ('1.weight', 'element', [0.4, 0.6])
    ]
    rule_dict = {
        '0.weight': [0.3, 0.5],


x = torch.rankd(2,3,4)
x_with_2n3_dimensions = x[1,:,:] 
scalar_x = x[1,1,1]

#numpy like slicing 
x = torch.rand(2,3)
print(x[:,1:]) #skipping first column

"""
A Simple Neural Network

Learning the PyTorch way of building a neural network is really important. 
IT is the most efficient and clean way of writting PyTorch code, and it also helps you to find
tutorials and sample snippets easy to follow, since they have the same structure
"""

def binary_encoder(input_size):
  def wrapper(num):
    ret = [int(i) for i in '{0:b}'.format(num)]
    return [0] * (input_size - len(ret)) + ret
  
def get_numpy_data(input_size=10, limit=1000):
  x = []
  y = []
  encoder = binary_encoder(input_size)
  for i in range(limit):
    x.append(encoder(i))
    if i % 15 == 0:
      y.append([1,0,0,0])
    elif i % 5 == 0:
      y.append([0,1,0,0])
  return training_test_gen(np.array(x), np.array(x))

"""
The endoer function encodes the input to binary number, which makes it easy for the neural network to learn.
"""

epochs = 500
batches = 64
lr = 0.01 
input_size = 10
output_size = 4
hidden_size = 100

for i in epoch:
  network_execution_over_whole_dataset()

"""
The learning rate decides how fast we eant our network to take feedbak from the error on each iteration.
It decides what to learn from the current iteration by forgetting what the network learned from all the previous iterations
"""

#Autograd 

x = torch.from_numpy(trX).type(dtype)
Y = torch.from_numpy(trY).type(dtype)
W1 = torch.randn(input_sise, hidden_sie, requires_grad=True).type(dtype)
W2 = torch.rnadn(hidden_size, output_size, required_grad=True).type(dtype)
b1 = torch.zeros(1, hidden_size, requires_grad=True).type(dtype)
b2 = torch.zeros(1, output_size, requires_grad=True).type(dtype)

prind(x.grad x.grad_fn, x)


for epoch in range(epochs):
  for batch in range(no_of_batches):
    start = batch * batches
    end = start + batches
    x_ = x[start:end]
    y_ = y[start:end]
    
    #build graph
    a2 = x_.matmul(w1)
    a2 = a2.add(b1)
    print(a2.grad, a2.grad_fn, a2)
    

class Linear(Module):
  def __init__(self, in_features, out_features, bias):
    super(Linear, sefl).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.Tensor(out_features, in_features)
