import numpy as np
import plot

class RBF:
    def __init__(self, mu, sig_squared):
        self.position = mu
        self.variance = sig_squared
    def phi(self, x):
        relative_position = x - self.position
        return np.exp(-relative_position*relative_position/(2*self.variance))

def run(weights, rbfs, x):
    return sum(w*rbf.phi(x) for w,rbf in zip(weights, rbfs))

def train(data, values, N = None):
    if N == None:
        N = len(data)
    mu = np.random.choice(data, N)
    sig = [min(a - b for b in data if a != b) for a in data]
    rbfs = [RBF(m, s) for m, s in zip(mu, sig)]
    phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in data])
    w = np.linalg.solve(phi, values)
    return w, rbfs

if __name__ == "__main__":
    data = list(np.arange(0, 6, .1))
    values = [np.sin(x) for x in data]
    w, rbfs = train(data, values, 10)
    result = [run(w, rbfs, x) for x in data]
    plot.plot(data, values)
    plot.plot(data, result)
