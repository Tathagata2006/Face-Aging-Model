import torch
import numpy as np

# empty uninitialised tensor creation
x = torch.empty(2, 3) 

# random tensor
x = torch.rand(2, 2)

#Zero Tensor + setting datatype
x = torch.zeros(1, 5, dtype = torch.double)

#tensor creation another method
x = torch.tensor([3, 5, 7])

#tensor operations
x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x+y #x-y, x*y, x/y
z = torch.add(x, y)  #torch.sub, torch.mul, torch.div
y.add_(x)  # inplace addition (y is modified)

x = torch.rand(5, 3)
print(x[1, 1])
print(x[1, 1].item())


print(x)
print(y)
print(z)


print(x)
print(x.dtype)          #prints data type
print(x.size())         #size of a tensor
 
#resizing tensors

a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
b = a.view(-1, 2)
print(b)

#torch tensor -> numpy arrays

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

print(type(a))
print(type(b))

a.add_(1)
print(a)
print(b)

#numpy arrays -> torch tensor

a = np.ones(6)
print(a)
b = torch.from_numpy(a)
print(b)

print(type(a))
print(type(b))

m = torch.ones(5, requires_grad=True)
print(m)

#identity matrix

a = torch.eye(4)
print(a)

