import numpy as np

#f= w*x

#f = 2*x

X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

w = 0.0

#model prediction
def forward(x):
    return w * x

#Losses
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

#gradient (dL/dw)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Model Training

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)
    
    #Loss
    l = loss(Y, y_pred)
    
    #gradient
    dw = gradient(X, Y, y_pred)
    
    #update weights
    w = w - learning_rate*dw
    
    if epoch%2 == 0:
        print(f'epoch {epoch+2}: w = {w:.3f}, l = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')


#USING PYTORCH

import torch

#f= w*x

#f = 2*x

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype = torch.float32, requires_grad=True)

#model prediction
def forward(x):
    return w * x

#Losses
def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')

#Model Training

learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    #prediction = forward pass
    y_pred = forward(X)
    
    #Loss
    l = loss(Y, y_pred)
    
    #gradient = backwardpass
    l.backward #dl/dw
    
    #update weights
    with torch.no_grad():
        w = w - learning_rate* float(w.grad)
    
    #zero gradients
    w.grad.zero_()    
    
    if epoch%2 == 0:
        print(f'epoch {epoch+2}: w = {w:.3f}, l = {l:.8f}')
        
print(f'Prediction after training: f(5) = {forward(5):.3f}')
