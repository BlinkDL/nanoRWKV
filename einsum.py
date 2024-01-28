import numpy as np
from einsum_cython import generalized_einsum

A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)

result = generalized_einsum(A, B, b'ij,ij->ijkl')
print(result)