import slam.lies as lies
import cv2
import numpy as np

rotation_vector = np.array([4, 1, 1], dtype=np.float64)

rotation_matrix = lies.so3exp(rotation_vector)
rotation_matirx2, _ = cv2.Rodrigues(rotation_vector)

print(rotation_matrix)
print(rotation_matirx2)

rotation_vector_back = lies.so3log(rotation_matrix)
rotation_vector_back2 = lies.so3log(rotation_matirx2)

print(rotation_vector_back)
print(rotation_vector_back2)
