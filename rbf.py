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

def least_squares(data, values, N = None):
    if N == None:
        N = len(data)
    mu = np.random.choice(data, N)
    sig = [min(a - b for b in data if a != b) for a in data]
    rbfs = [RBF(m, s) for m, s in zip(mu, sig)]
    phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in data])
    w = np.linalg.solve(phi, values)
    return w, rbfs

def iterative_batch(train_data, values, error=0.1, N = None):
    if N == None:
        N = len(train_data)
    w = np.random.normal(-1, 1, N)
    mu = np.random.choice(train_data, N)
    sig = [min(a - b for b in train_data if a != b)**2 for a in train_data]
    rbfs = [RBF(m, s) for m, s in zip(mu, sig)]
    big_phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in train_data ])
    total_error =  sum((big_phi.dot(w) - values) ** 2)

    w = np.linalg.lstsq(big_phi.T.dot(big_phi), big_phi.T.dot(values), rcond=-1)[0]
    total_error =  sum((big_phi.dot(w) - values) ** 2)
    return w, rbfs

if __name__ == "__main__":
    train_data = list(np.arange(0, 2*np.pi, .1))
    test_data = list(np.arange(0.05, 2*np.pi, .1))
    values = np.array([np.sin(x) for x in train_data])
    w, rbfs = iterative_batch(train_data, values, N=5)
    # w, rbfs = least_squares(train_data, values)
    result = [run(w, rbfs, x) for x in test_data]
    plot.plot(train_data, values, [-1, 1])
    plot.plot(test_data, result, [-1, 1])