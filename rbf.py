import numpy as np
import plot
import math as m

class RBF:
    def __init__(self, mu, sig_squared):
        self.position = mu
        self.variance = sig_squared
    def phi(self, x):
        distance = np.linalg.norm(x - self.position)**2
        return np.exp(-distance/(2*self.variance))

def run(weights, rbfs, x):
    return sum(w*rbf.phi(x) for w,rbf in zip(weights, rbfs))

def run_all(weights, rbfs, data):
    return [run(weights, rbfs, x) for x in data]

def least_squares(data, values, N = None):
    if N == None:
        N = len(data)
    mu = np.random.choice(data, N)
    sig = [min(a - b for b in data if a != b) for a in data]
    rbfs = [RBF(m, s) for m, s in zip(mu, sig)]
    phi = np.array([[rbf.phi(x) for rbf in rbfs] for x in data])
    w = np.linalg.solve(phi, values)
    return w, rbfs

def delta_epoch(data, answer, w, rbfs, eta):
    for pt, desired in zip(data, answer):
        phi_vector = np.array([rbf.phi(pt) for rbf in rbfs])
        difference = desired - np.transpose(phi_vector).dot(w)
        left = phi_vector.reshape((-1, 1))
        right = difference.reshape((1, -1))
        w += eta*np.dot(left, right).reshape(w.shape)

def delta(data,answer,w,rbfs,eta,epoch):
    error_array = []
    for _ in range(epoch):
        delta_epoch(data, answer, w, rbfs, eta)
        error_array += [error(answer,run_all(w,rbfs,data))]
    return w,error_array

def error(answer,result):
    return sum(abs(a-r) for a, r in zip(answer, result))/len(answer)

def vanilla(rbfs,data,iterations):
    for j in range(0,iterations):
        winner = rbfs[0]
        temp = 100
        step =0.01
        minimum = 10
        point = np.random.choice(data)
        for i in range(0,len(rbfs)):
            temp = abs(rbfs[i].position - point)
            if temp<minimum:
                minimum = temp
                winner = rbfs[i]
                winner_num=i
        rbfs[winner_num].position+= step*(point-rbfs[winner_num].position)
    return rbfs

def non_vanilla(rbfs,data,iterations):
    for j in range(0,iterations):
        winner = rbfs[0]
        print("wiener", winner)
        temp = 100
        eta =0.01
        minimum = 10
        point = np.random.choice(data)
        for i in range(0,len(rbfs)):
            temp = abs(rbfs[i].position-point)
            if temp<minimum:
                minimum = temp
                winner = rbfs[i]
                winner_num=i
                delta_pos =eta*(point-rbfs[winner_num].position)
        rbfs[winner_num].position+=  delta_pos
        for i in range(0,len(rbfs)):
            rbfs[i].position += 0.1* delta_po
    mu = np.random.choice(data,N)
    return rbfs


def rbfs_init(data,rng,N=5):
    return [RBF(i*(rng/N),0.1) for i in range(N)]

def rbfs_init_rn(data,rng,N=None):
    mu = np.random.choice(data,N)
    rbfs=np.empty([N,1],RBF)
    for i in range(0,N):
        rbfs[i]=RBF(mu[i],0.1)
    return rbfs



if __name__ == "__main__":
    epoch=40
    rbf_count=10
    error_array_acc=np.zeros(epoch)
    train_data = list(np.arange(0, 2*np.pi, .1))
    test_data = list(np.arange(0.05, 2*np.pi, .1))
    values = np.array([np.sin(2*x) for x in train_data])
    for i in range(5):
        noise = np.random.normal(0,0.1,len(train_data))
        train_data+=noise
        mu = np.random.choice(train_data, len(train_data))
        rbfs = rbfs_init(train_data,2*np.pi,rbf_count)
        rbfs =vanilla(rbfs,train_data,1000)
        for i in range(0,len(rbfs)):
            print("rbf  ", i," ",rbfs[i].position )
        w = np.ones((len(rbfs)))
        w = w*0.1
        w,error_array = delta(train_data,values,w,rbfs,0.04,epoch)
        # w, rbfs = least_squares(train_data, values)
        result = [run(w, rbfs, x) for x in test_data]
        error_array_acc = [b + a/5 for a, b in zip(error_array, error_array_acc)]
    plot.plot_points(error_array_acc)
    print(values, result)
    print(error(values,result))
    plot.plot(train_data, values, [-1, 1])
    plot.plot(test_data, result, [-1, 1])
