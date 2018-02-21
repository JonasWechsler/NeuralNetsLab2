import numpy as np
import plot

class RBF:
    def __init__(self, mu, sig_squared):
        self.position = mu
        self.variance = sig_squared
    def phi(self, x):
        distance = np.absolute(x - self.position)**2
        return np.exp(-distance/(2*self.variance))

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

def iterative_batch(train_data, test_data, train_values, test_values, error=0.1, N = None):
    if N == None:
        N = len(train_data)
    w = np.random.normal(-1, 1, N)
    mu = np.random.choice(train_data, N)
    sig = [min(a - b for b in train_data if a != b)**2 for a in train_data]
    rbfs = [RBF(m, s) for m, s in zip(mu, sig)]
    big_phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in train_data])

    w = np.linalg.lstsq(big_phi.T.dot(big_phi), big_phi.T.dot(train_values), rcond=-1)[0]

    # Test data error
    big_phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in test_data])
    total_error =  sum((big_phi.dot(w) - test_values) ** 2)/len(test_data)
    return w, rbfs, total_error

def least_squares_sin_noise(N=5, shouldPlot=True):
    train_data = list(np.arange(0, 2*np.pi, 0.1))
    test_data = list(np.arange(0.05, 2*np.pi, 0.1))
    train_noise = np.random.normal(0,0.1,len(train_data))
    test_noise = np.random.normal(0,0.1,len(test_data))
    train_values = np.array([np.sin(2*x) for x in train_data])
    test_values = np.array([np.sin(2*x) for x in test_data])
    train_values += train_noise
    test_values += test_noise
    w, rbfs, error = iterative_batch(train_data, test_data, train_values, test_values, N=N)
    result = [run(w, rbfs, x) for x in test_data]
    if shouldPlot:
        plot.plot(train_data, train_values, [-1, 1])
        plot.plot(test_data, result, [-1, 1])
    return error

def least_squares_square(N=5, shouldPlot=True):
    train_data = list(np.arange(0, 2*np.pi, 0.1))
    test_data = list(np.arange(0.05, 2*np.pi, 0.1))
    train_values = np.array([1 if x < np.pi/2 or (x>np.pi and x<3*np.pi/2) else -1 for x in train_data])
    test_values = np.array([1 if x < np.pi/2 or (x>np.pi and x<3*np.pi/2) else -1 for x in test_data])
    w, rbfs, error = iterative_batch(train_data, test_data, train_values, test_values, N=N)
    result = [run(w, rbfs, x) for x in test_data]
    if shouldPlot:
        plot.plot(train_data, train_values, [-1, 1])
        plot.plot(test_data, result, [-1, 1])
    return error

def least_squares_square_modified(N=5, shouldPlot=True):
    train_data = list(np.arange(0, 2*np.pi, 0.1))
    test_data = list(np.arange(0.05, 2*np.pi, 0.1))
    train_values = np.array([1 if x < np.pi/2 or (x>np.pi and x<3*np.pi/2) else -1 for x in train_data])
    test_values = np.array([1 if x < np.pi/2 or (x>np.pi and x<3*np.pi/2) else -1 for x in test_data])
    w, rbfs, error = iterative_batch(train_data, test_data, train_values, test_values, N=N)
    result = [run(w, rbfs, x) for x in test_data]
    result = [1 if x>0 else -1 for x in result]
    error =  sum((result - test_values) ** 2)/len(test_data)
    if shouldPlot:
        plot.plot(train_data, train_values, [-1, 1])
        plot.plot(test_data, result, [-1, 1])
    return error

def least_squares_sin(N=5, shouldPlot=True):
    train_data = list(np.arange(0, 2*np.pi, 0.1))
    test_data = list(np.arange(0.05, 2*np.pi, 0.1))
    train_values = np.array([np.sin(2*x) for x in train_data])
    test_values = np.array([np.sin(2*x) for x in test_data])
    w, rbfs, error = iterative_batch(train_data, test_data, train_values, test_values, N=N)
    result = [run(w, rbfs, x) for x in test_data]
    if shouldPlot:
        plot.plot(train_data, train_values, [-1, 1])
        plot.plot(test_data, result, [-1, 1])
    return error

if __name__ == "__main__":
    num = 100
    test_nodes = [_ for _ in range(1, num)]
    result = []
    for n in test_nodes:
        result.append(least_squares_sin(n, False))
        #result.append(least_squares_square(n, False))
        #result.append(least_squares_sin_noise(n, False))
        #result.append(least_squares_square_modified(n, False))

    plot.plot(test_nodes, result, [0, 1])