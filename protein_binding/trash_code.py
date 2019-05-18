import numpy as np
import csv
import matplotlib.pyplot as plt

'''
L = []
with open('data/tensor/log', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        L.append(row)

arr = np.array(L, dtype=float)
print(np.mean(arr, axis=0))
plt.hist(arr[:, 0], bins=100)
plt.hist(arr[:, 1], bins=100)
plt.hist(arr[:, 2], bins=100)
plt.show()

print(sum(arr[:, 0] > 42)/ len(arr))
print(sum(arr[:, 1] > 32)/ len(arr))
print(sum(arr[:, 2] > 32)/ len(arr))

a grid of 42*32*32  = 43008 keeps approx every points of the extracted PDBs 
PCA enables to avoid using a square grid
'''