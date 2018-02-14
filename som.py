import numpy as np
import plot
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class SOM:
    def __init__(self, width, height, default=np.array([0,0])):
        self.width = width
        self.height = height
        self._lattice = [default]*width*height
    def set(self, x, y, vector):
        self._lattice[y*self.width+x] = vector
    def get(self, x, y):
        return self._lattice[y*self.width+x]
    def _in_bounds(self, x, y):
        return x >= 0 and y >= 0 and x < self.width and y < self.height
    def _adjacent(self, x, y, radius):
        return [(i, j) for i in range(x-radius, x+radius) for j in range(y-radius, y+radius) if self._in_bounds(i, j)]
    def get_adjacent(self, x, y, radius):
        return [self.get(i, j) for i, j in self._adjacent(x, y, radius)]
    def get_all(self, with_position = False):
        result = []
        if with_position:
            for x in range(self.width):
                for y in range(self.height):
                    result += [(self.get(x, y), x, y)]
            return result
        return self._lattice
    def get_connections(self):
        result = []
        for x in range(self.width):
            for y in range(self.height):
                if self._in_bounds(x+1, y):
                    result += [(self.get(x, y), self.get(x+1, y))]
                if self._in_bounds(x, y+1):
                    result += [(self.get(x, y), self.get(x, y+1))]
        return result


def train(data, som, neighborhood_radius=1, nu=0.2):
    _, x, y, winner = min((np.linalg.norm(data-node), x, y, node) for node, x, y in som.get_all(True))
    neighborhood = som.get_adjacent(x, y, neighborhood_radius)
    for neighbor in neighborhood:
        neighbor += nu*(data-neighbor)


if __name__ == "__main__":
    som = SOM(10, 10)
    for i in range(som.width):
        for j in range(som.height):
            som.set(i, j, np.array([np.float(i-som.width/2)*.6, np.float(j-som.width/2)*.6]))
    data = []
    for i in np.arange(0, 2*np.pi, 0.1):
        data += [np.array([np.cos(i)*5, np.sin(i)*5])]
        data += [np.array([np.cos(i)*6, np.sin(i)*6])]
        data += [np.array([np.cos(i)*7, np.sin(i)*7])]
    for _ in range(100):
        for d in data:
            train(d, som, 3)
    plot.plot_lattice(som.get_connections())
