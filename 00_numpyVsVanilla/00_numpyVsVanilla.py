import numpy as np
import time
import math

#reverse vector
a = np.array([10,20,30])
print(np.flip(a,0))

#product of the element on its diagonal
a = np.array([[1,3,8],[-1,3,0],[-3,9,2]])
print(np.prod(np.diagonal(a)))

#random vector (3,6)
a = np.random.rand(3,6)
print(a, a.mean())

#how many times element in a is higher than the corrisponding element of b
#should be 6
a = np.array([[1,5,6,8],[2,-3,13,23],[0,-10,-9,7]])
b = np.array([[-3,0,8,1],[-20,-9,-1,32],[7,7,7,7]])
print(((a > b) == True).sum())

#create and normalize the following matrix
a = np.array([[0.35, -0.27, 0.56],[0.15,0.65, 0.42],[0.73,-0.78,-0.08]])
print((a-a.min())/(a.max()-a.min()))


#numpy vs vanilla performances
N = 1000  # number of rows (or the amount of examples of the dataset)
M = 30  # number of columns (the number of features of each example)

A = np.random.rand(N, M) #matrice A (NxM)
b = np.random.rand(M) #vector b (M elementi)

#print('A:',A,A.shape)
#print('b:',b, b.shape)

#distanza euclidea tra b e ogni riga di A (Ai)

def compute_distances(A: np.ndarray, b: np.ndarray): #typing  N-dimensional array
    N, M = A.shape

    assert len(b.shape) == 1 and b.shape[0] == M
    distances = np.zeros((N,))

    for i in range(N):
        sum = 0
        for j in range(M):
            sum += (A[i][j] - b[j])**2
        distances[i] = math.sqrt(sum)
    return distances


def compute_distances_with_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray: #typing, They can be used by third party tools such as type checkers, IDEs, linters, etc.
    N, M = A.shape
    assert len(b.shape) == 1 and b.shape[0] == M
    distances = np.sqrt(np.sum((A - b)**2, axis=1)) #versione 1
    #distances2 =  np.sqrt(np.power((A - b[np.newaxis, :]), 2).sum(1)) #versione 2
    #print(distances[0:5])
    #print(distances2[0:5])
    #assert np.allclose(distances, distances2)
    #print(type(distances))
    return distances

start = time.time()
d1 = compute_distances(A, b)
end = time.time()
print("Pythonic Euclidean distance, time: ", end - start)

start = time.time()
d2 = compute_distances_with_numpy(A, b)
end = time.time()
print("Numpythonic Euclidean distance, time: ", end - start)

assert np.allclose(d1, d2)
