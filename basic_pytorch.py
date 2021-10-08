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
