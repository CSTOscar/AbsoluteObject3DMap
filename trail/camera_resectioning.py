import numpy as np
import cv2

xyz = np.asmatrix([[1], [1], [1], [1]])

id_proj = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
id_mat = np.asmatrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
print(id_mat)

uv = id_mat @ xyz
print(uv)
print(np.invert(id_mat))
id_mat_inv = np.linalg.pinv(id_mat)
print(id_mat_inv)
print(id_mat_inv @ uv)

R = np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
T = np.asmatrix([[1], [1], [1]])

RconT = np.concatenate((R, T), axis=1)

zeros = np.asmatrix(np.zeros([1, 4]))

print(RconT)
print(zeros)

E = np.concatenate((RconT, zeros), axis=0)
print(E)

mat = np.array([1, 2, 3], dtype=np.float64)
x = cv2.Rodrigues(mat)
print(x)
