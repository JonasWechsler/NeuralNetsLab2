import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

def plot(x_values, y_values):
    plt.plot(x_values, y_values)
    plt.show()

def plot_points(y_values):
    x_values = list(range(len(y_values)))
    plt.plot(x_values, y_values)
    plt.show()
