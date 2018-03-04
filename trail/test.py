import numpy as np
from sklearn import mixture

#
# arr = np.array([[(i * 10 + j) for j in range(10)] for i in range(10)])
# print(arr[1:5, 1:3])
#
# a = [i for i in range(10)]
#
# print(list(map(lambda e: e * 2, filter(lambda e: e % 2 == 0, a))))
#
# A = np.asmatrix([[1], [2], [3], [4]])
# B = np.asmatrix([[4], [3], [2], [1]])
#
# print(A.T @ B > 0)
#

#
# print(pos.shape)
# print(type(pos))
# print(np.array(pos).flatten())
# print(type(pos))
#
# P1 = np.random.multivariate_normal([15, 2, 33], [[0.011 if i == j else 0 for j in range(3)] for i in range(3)], 23)
# P2 = np.random.multivariate_normal([2, 23, 2], [[0.011 if i == j else 0 for j in range(3)] for i in range(3)], 50)
# print(P1.shape)
# print(P2.shape)
# P = np.concatenate((P1, P2), axis=0)
#
# projection_position = P
#
# projection_mixture = mixture.BayesianGaussianMixture(n_components=5,
#                                                      covariance_type='full')
# projection_mixture.fit(projection_position)
# projection_mixture.predict(projection_position)
# print(len(projection_position), 'size', len(projection_mixture.means_))
#
# print(projection_mixture.converged_)
# print(projection_mixture.means_)
# print(projection_mixture.weights_)
# print(projection_mixture.get_params())
#
# print(projection_mixture.predict(P))

pos_tup = (1, 2, 3)
pos = np.asmatrix(np.array(pos_tup).reshape((3, 1)))
print(pos)
print(type(pos))
print(type(pos) == np.matrix)
print(pos_tup)
print(type(pos_tup))
print(type(pos_tup) == tuple)

lst = [pos_tup]

for ii in lst:
    ii = (123, 32)

print(lst[0])
print(isinstance(pos, np.matrix))
print(type(lst))
print(type(np.asarray([1])))
a = list((pos_tup, pos))
print(a)


def ret_test():
    ret = 1
    return ret


result = list(ret_test())

print(result[0])

print((1, 1) + (1, 1))
