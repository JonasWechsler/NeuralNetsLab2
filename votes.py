import plot
import numpy as np
from som import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def read_additional_file(inf_name, f=int):
    with open("data/{}".format(inf_name)) as inf:
        return [f(r.strip()) for r in inf.readlines() if r[0] != '%' and r.strip() != '']
if __name__ == "__main__":
    with open("data/votes.dat") as inf:
        data_list = list(map(float, inf.read().split(",")))
        data_array = np.array(data_list).reshape((349, 31))
    party_list = read_additional_file("mpparty.dat")
    sex_list = read_additional_file("mpsex.dat")
    district_list = read_additional_file("mpdistrict.dat")
    names_list = read_additional_file("mpnames.txt", str)
    som = SOM(10, 10)
    for i in range(som.width):
        for j in range(som.height):
            som.set(i, j, np.array([np.random.uniform() for _ in range(31)]))
    for radius in [10, 8, 6, 4, 2, 0]:
        for epoch in range(4):
            for member in data_array:
                train(member, som, radius)
    pp = PdfPages("out/votes.pdf")
    for labels, name, width in [(party_list, "party", 4), (sex_list, "sex", 4), (district_list, "district", 4)]:
        for label in set(labels):
            plt.figure(figsize=(width, width))
            plt.title("{}:{}".format(name, label))
            print(name, label)
            result = np.zeros((10, 10))
            for idx, member in enumerate(data_array):
                i, j = index(member, som)
                #print(i, j, names_list[idx])
                if labels[idx] == label:
                    result[i][j] += 1
            plt.imshow(result, cmap='cool', interpolation='nearest')
            pp.savefig()
            plt.close()
    pp.close()
