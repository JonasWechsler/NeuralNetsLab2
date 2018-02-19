import  numpy as np
from som import *


if __name__ == "__main__":
    with open("data/animals.dat") as inf:
        data_list = list(map(int, inf.read().split(",")))
        data_array = np.array(data_list).reshape((32, 84))
    with open("data/animalnames.txt") as inf:
        data_labels = inf.read().replace("\'","").replace("\t", "").split("\n")
    assert(data_array.shape == (32, 84))
    assert(len(data_labels) == 32)
    som = SOM(100, 1)

    #32 animal species and 84 attributes
    print(data_array)
    print(data_labels)

    for i in range(som.width):
        for j in range(som.height):
            som.set(i, j, np.array([np.random.uniform() for _ in range(84)]))

    for radius in [40, 30, 20, 10, 0]:
        for epoch in range(4):
            for animal in data_array:
                train(animal, som, radius)
    for idx, animal in enumerate(data_array):
        print(idx, data_labels[idx], index(animal, som))
    result = sorted([(index(animal, som), data_labels[idx]) for idx, animal in enumerate(data_array)])
    print(result)
    with open("out/animals.out", "w") as outf:
        for r in result:
            outf.write(r[1])
            outf.write("\n")
    
    
