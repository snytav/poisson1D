# -*- coding: utf-8 -*-
"""Копия блокнота "PDE-Poisson.ipynb"

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aL6xz_T-tstK6WQ9NtoeJBBR5Q6oDQ3t
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#from my_optimizer import optimizer_step

from matplotlib import pyplot as plt
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
   x = torch.ones(20)
   print(x.device)
   x = x.to("cuda:0")
   # or x=x.to("cuda")
   print(x)
   print(x.device)

from auxiliary import f

from PDE import PDEnet


nx = 10
ny = nx
pde = PDEnet(nx)
dx = 1. / nx
dy = 1. / ny

x_space = torch.linspace(0, 1, nx)
y_space = torch.linspace(0, 1, ny)
print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
  print(torch.device)
  print(x_space.device)
  x_space.to("cuda:0")
  #y_space.to("cuda:0")
  #print(x_space)
  print(x_space.device)

input_point = torch.zeros(2)
net_out = pde.forward(input_point)
from auxiliary import psy_trial


lmb = 0.001
N_epoch = 100
from train_func import train
train(lmb,x_space,y_space,N_epoch,pde,psy_trial,f)

from compare import compare_to_analytic
from analytics import analytic_solution
mape,md,where_md = compare_to_analytic(x_space,y_space,psy_trial,analytic_solution,pde)





