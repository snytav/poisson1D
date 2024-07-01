import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def compare_to_analytic(x_space,psy_trial,analytic_solution,pde):
    nx = x_space.shape[0]
    surface    = np.zeros(nx)
    an_surface = np.zeros(nx)
    for i, x in enumerate(x_space):
            input_point = torch.Tensor([x])
            input_point.requires_grad_()
            net_out = pde.forward(input_point)

            psy_t = psy_trial(input_point, net_out)
            surface[i] = psy_t
            an_surface[i] = analytic_solution(x)
    diff = np.max(np.abs(surface-an_surface))
    mape = mean_absolute_percentage_error(an_surface, surface)
    plt.figure()
    plt.plot(x_space,surface,'o',label='neural',color='red')
    plt.plot(x_space,an_surface,label='analytic',color='green')
    plt.legend()
    diff = np.abs(an_surface - surface)
    md = np.max(diff)
    where_md = np.where(diff == md)
    return mape,md,where_md

if __name__ == '__main__':
    x_space = np.linspace(0,1.0,10)
    y_space = x_space
    X,Y = np.meshgrid(x_space,y_space)
    an_sol = np.sin(X)*Y**2
    from auxiliary import psy_trial
    from analytics import analytic_solution
    from PDE import PDEnet
    pde = PDEnet(x_space.shape[0])
    df = compare_to_analytic(x_space, y_space, psy_trial, analytic_solution, pde)
    qq = 0
