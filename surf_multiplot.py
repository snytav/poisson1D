# -*- coding: utf-8 -*-
"""surf_multiplot

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mKaHTA8vCgU3J9l1MzeJD9rUzLEE0Wub
"""

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z1 = np.sin(R)
Z2 = np.sin(R)


def plot_2_3D_figures(X,Y,Z1,Z2,title1,title2):

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(0.5))

    # =============
    # First subplot
    # =============
    # set up the Axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # plot a 3D surface like in the example mplot3d/surface3d_demo

    surf1 = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    fig.colorbar(surf1, shrink=0.5, aspect=10)
    ax.set_title(title1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # ==============
    # Second subplot
    # ==============
    # set up the Axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    # plot a 3D wireframe like in the example mplot3d/wire3d_demo
    surf2 = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.coolwarm,
                          linewidth=0, antialiased=False)
    fig.colorbar(surf2, shrink=0.5, aspect=10)
    ax.set_title(title2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

if __name__ == '__main__':
     plot_2_3D_figures(X,Y,Z1,Z2,'z1','z2')