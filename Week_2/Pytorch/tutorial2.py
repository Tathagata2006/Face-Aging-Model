import torch
import numpy as np

x = torch.rand(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
z = z.mean()
print(z)

#computing derivatives dy/dx, dz/dx

z.backward() #dz/dx
print(x.grad)

z = y*y*2
v = torch.tensor([1, 1, 0.001], dtype = torch.float32)
z.backward(v)
print(x.grad)

#when we need to use x as without any grad require

#x.requires_grad_(False)
#y = x.detach()
#with torch.no_grad() :


#Model Training (Dummy Example)

weights = torch.ones(4, requires_grad=True)

for epoch in range(2) :
    model_output = (weights*weights*6).sum()
    
    model_output.backward()
    
    print(weights.grad) 
    weights.grad.zero_()
    
    