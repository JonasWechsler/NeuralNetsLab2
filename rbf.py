import numpy as np
import plot
import math as m

class RBF:
    def __init__(self, mu, sig_squared):
        self.position = mu
        self.variance = sig_squared
    def phi(self, x):
        relative_position = x - self.position
        return np.exp(-relative_position*relative_position/(2*self.variance))

class RBF_2D:
    def __init__(self, x,y ,sig_squared):
        self.x=x
        self.y=y
        self.variance = sig_squared
    def phi(self, x, y):
        relative_x = x - self.x
        relative_y = y - self.y
        return np.exp(-sqrt(relative_y**2+relative_x**2)**2/(2*self.variance))
        

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

def delta(data,answer,w,rbfs,eta,epoch):
    error_array =np.zeros(epoch)
    epoch_array = np.zeros(epoch)
    for i in range(epoch):
        for j in range(len(data)):
            phi_vector= np.array([rbf.phi(data[j]) for rbf in rbfs])
            #print("answer[j]",answer[j], "np.transpose(phi_vector)",np.transpose(phi_vector),"w",w,"result",answer[j] - np.transpose(phi_vector).dot(w))
            w += eta*(answer[j] - np.transpose(phi_vector).dot(w))*phi_vector
        error_array[i]=error(answer,run(w,rbfs,data))
        epoch_array[i]=i
    return w,error_array,epoch_array

def error(answer,result):
    err =0
    for i in range(len(answer)):
        err += abs(answer[i]-result[i])
    return err/len(answer)


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
    mu = np.random.choice(data,N)s
    return rbfs


def rbfs_init(data,rng,N=None):
    rbfs=[RBF(0,0) for i in range(0,N)]
    for i in range(0,N):
        rbfs[i]=RBF(0+i*(rng/N),0.1)
    return rbfs

def rbfs_init_2d(data,y_rng,x_rng,N=None):
    rbfs=[RBF_2d(0,0,0) for i in range(0,N)]
    for i in range(0,N):
        rbfs[i]=RBF(0+i*(x_rng/N),0+i*(y_rng/N,0.1))
    return rbfs

def rbfs_init_rn(data,rng,N=None):
    mu = np.random.choice(data,N)
    rbfs=np.empty([N,1],RBF)
    for i in range(0,N):
        rbfs[i]=RBF(mu[i],0.1)
    return rbfs



if __name__ == "__main__":
    epoch=2000
    error_array_acc=np.zeros(epoch)
    train_data = list(np.arange(0, 2*np.pi, .1))
    test_data = list(np.arange(0.05, 2*np.pi, .1))
    values = np.array([np.sin(2*x) for x in train_data])
    for i in range(0,5):
        noise = np.random.normal(0,0.1,len(train_data))
        train_data+=noise
        mu = np.random.choice(train_data, len(train_data))
        rbfs = rbfs_init(train_data,2*np.pi,5)
        rbfs =vanilla(rbfs,train_data,1000)
        for i in range(0,len(rbfs)):
            print("rbf  ", i," ",rbfs[i].position )
        w = np.ones((len(rbfs)))
        w = w*0.1
        w,error_array,epoch_array = delta(train_data,values,w,rbfs,0.04,epoch)
        # w, rbfs = least_squares(train_data, values)
        result = [run(w, rbfs, x) for x in test_data]
        error_array_acc = [sum(x)/5 for x in zip(error_array, error_array_acc)]
    plot.plot(epoch_array,error_array_acc)
    print(error(values,result))
    plot.plot(train_data, values, [-1, 1])
    plot.plot(test_data, result, [-1, 1])
