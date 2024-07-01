import torch
import numpy as np

def A(x):
    return (0.5*x[0]**2)

def psy_trial(x, net_out):
    return A(x) + x[0] * (1 - x[0])  * net_out


def f(x):
    return 1.0