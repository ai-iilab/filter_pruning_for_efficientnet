"""
One of the core philosophes of PyTorch, which came about with the evolution of PyTorch it self, is innteroperability.
The development teram invested a lot of time into enabling interoperability between different frameworks, such as ONNX, DLPak, and so on. 
Examples of these will be shown in latyer chapters, but here we will discuss how the internals of PyTorch are designed to accommodate this requirement without
compromising on speed
"""

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
