"""
Zhang, F., Ono, N., & Kanaya, S. (2025). 
Interpret Gaussian process models by using integrated gradients. 
Molecular Informatics, 44(1), e202400051.

https://github.com/bbbccc88/Interpret-Gaussian-Process-Models-by-using-Integrated-Gradients.git

MIT License

Copyright (c) 2024 bbbccc88

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random
import numpy as np
import torch
from tqdm import tqdm
import joblib


def prepare_for_grad(x, delta=1e-3):
    
    """
    Args:
        x : input(n,)
        delta : small amount
        
    Return:
        prepare_for_grad_x : data for calculate grads (2n, n)
    """
    
    x_for_grad_list = []
    
    for i in range(x.shape[0]):
        tmp = torch.vstack((x, x))
        tmp[0,i] += delta
        tmp[1,i] -= delta
        x_for_grad_list.append(tmp)
        
    prepare_for_grad_x = torch.cat(x_for_grad_list, dim=0)
        
    assert x.shape[0]*2 == prepare_for_grad_x.shape[0], f'{x.shape[0]*2}→{prepare_for_grad_x.shape[0]}'
    assert x.shape[0] == prepare_for_grad_x.shape[1], f'{x.shape[0]}→{prepare_for_grad_x.shape[1]}'
    
    return prepare_for_grad_x

def calc_grad(y, x):
    
    """
    Args:
        y : target variables (2n,)
        x : input varibales (2n, n)

    Return:
        grad : (n)
    """
    grad_list = []
    
    for i in range(x.shape[1]):
        grad = (y[2*i] - y[2*i+1])/(x[2*i, i] - x[2*i+1, i])
        grad_list.append(grad)
        
    assert x.shape[1] == torch.tensor(grad_list).shape[0], f'{x.shape[1]}→{torch.tensor(grad_list).shape[0]}'
        
    return torch.tensor(grad_list)

def IG_torch(model, x, N=100):
    
    """
    Args:
        model : trained gaussian process model
        x : input varibales (1, n)
        N : the number of steps for calculate integrated gradients

    Return:
        IG : integrated gradients for interpret predicted mean by using automatic differentiation
    """
    
    IG_tmp = []
    
    for n in range(N+1): 
        x_n = x * n/N
        x_n = x_n.detach().clone().requires_grad_(True)  
        f_preds = model(x_n.reshape(1,-1))
        f_preds_mean = f_preds.mean
        f_preds_mean.backward()
        IG_tmp.append(x_n.grad.clone())
    
    IG_tmp = torch.stack(IG_tmp, dim=0)
    IG = torch.mean(IG_tmp, dim=0).detach()
    
    return IG

def GP_IG(model, x, N=100, sampling=100):
    
    """
    Args:
        model : trained gaussian process model
        x : input varibales (n)
        N : the number of steps for calculate integrated gradients
        samping : the number of functions sampled from model

    Return:
        all_IG : all integrated gradients for interpret predicted mean and std. by using numerical differentiation
    """
    
    for s in range(sampling):
        
        # create data necessary for path integration
        for n in range(N+1):
            x_n = x * n/N
            x_n_for_grad = prepare_for_grad(x_n)
            
            if n == 0:
                all_x_n_for_grad = x_n_for_grad.detach().clone()
            else:
                all_x_n_for_grad = torch.vstack((all_x_n_for_grad, x_n_for_grad))
                
        assert all_x_n_for_grad.cpu().numpy().shape == (x.shape[0]*2*(N+1) , x.shape[0]), f"{all_x_n_for_grad.numpy().shape}→{(x.shape[0]*2*(N+1) , x.shape[0])}"
        
        # sampling latent function
        all_f_sampling_for_grad = model(all_x_n_for_grad).sample()
        
        # calculate gradients at each path
        for n2 in range(N+1):
            
            f_sampling_for_grad = all_f_sampling_for_grad[n2*x.shape[0]*2:((n2+1)*x.shape[0]*2)]
            x_n_for_grad = all_x_n_for_grad[n2*x.shape[0]*2:((n2+1)*x.shape[0]*2),:]

            # calculate
            grad = calc_grad(f_sampling_for_grad, x_n_for_grad).detach()
            
            if n2 == 0:
                all_grad = grad.detach().clone()
            else:
                all_grad = torch.vstack((all_grad, grad))

        IG = all_grad.mean(axis=0).detach()
        
        if s == 0:
            all_IG = IG.detach().clone()
        else:
            all_IG = torch.vstack((all_IG, IG))
        
    assert all_IG.cpu().numpy().shape == (sampling, x.shape[0]), f"{all_IG.numpy().shape }→{(sampling, x.shape[0])}"
    
    return all_IG