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
