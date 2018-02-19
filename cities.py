import numpy as np
from som import *

if __name__ == "__main__":
    with open("data/cities.dat") as inf:
        content = inf.readlines()
        data = []
        for line in content:
            if ',' in line and line[0] != '%':
                data += line.split(',')
        data = [line.replace(';','').strip() for line in data]
        data = list(map(float, data))
        data = np.array(data).reshape((-1,2))
    som = SOM(10, 1, loop=True)
    for i in range(som.width):
        for j in range(som.height):
            som.set(i, j, np.array([np.random.uniform() for _ in range(2)]))
    plot.plot_lattice(som.get_connections(), False)
    plot.plot(data.T[0], data.T[1])
    for radius in [2, 1, 0]:
        for epoch in range(7):
            for pt in data:
                train(pt, som, radius)
    for idx, pt in data:
        print(idx, pt, index(pt, som))
    plot.plot_lattice(som.get_connections(), False)
    plot.plot(data.T[0], data.T[1])
