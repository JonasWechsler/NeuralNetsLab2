from rbf import *
import numpy as np

def rbfs_init_2d(y_rng,x_rng,N=5):
    return [RBF(np.array([x_rng*(i/N), y_rng*(j/N)]), 0.1) for i in range(N) for j in range(N)]

if __name__ == "__main__":
    epoch=500
    error_array_acc=np.zeros(epoch)
    train_data_input = []
    train_data_output = []
    with open("data/ballist.dat") as inf:
        lines = [line.split("\t") for line in inf.readlines()]
        for line in lines:
            if len(line) < 2:
                continue
            inp, out = line[0], line[1]
            in_pt = np.array(list(map(float,inp.split(" "))))
            out_pt = np.array(list(map(float,out.split(" "))))
            train_data_input += [in_pt]
            train_data_output += [out_pt]
    train_data_output = np.array(train_data_output)

    for _ in range(1):
        rbfs = rbfs_init_2d(1, 1, 5)
        for i in range(0,len(rbfs)):
            print("rbf  ", i," ",rbfs[i].position )
        w = np.array([[np.random.uniform()*0.1, np.random.uniform()*0.1] for _ in rbfs])
        print(w)
        w, error_array = delta(train_data_input,train_data_output,w,rbfs,0.04,epoch)
        result = [run(w, rbfs, x) for x in train_data_input]
        plot.plot_points(error_array)
    data = [[run(w, rbfs, [x, y])[0] for x in np.arange(0, 1, .1)] for y in np.arange(0, 1, .1)]
    plot.plot_heatmap(data)
    data = [[run(w, rbfs, [x, y])[1] for x in np.arange(0, 1, .1)] for y in np.arange(0, 1, .1)]
    plot.plot_heatmap(data)
    #plot.plot(train_data_input, train_data_output, [-1, 1])
    #plot.plot(train_data_input, result, [-1, 1])
