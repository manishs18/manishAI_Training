import numpy as np
print('NumPy version:', np.__version__)

# From a Python list
a = np.array([1, 2, 3, 4, 5])
print('Array:', a)
print('Shape:', a.shape)      # (5,) - 1-D, 5 elements
print('Dtype:', a.dtype)      # int64 by default
print('Ndim:', a.ndim)        # 1

# 2-D array (matrix)
mat = np.array([[1, 2, 3],
                [4, 5, 6]])
print('Shape:', mat.shape)    # (2, 3) - 2 rows, 3 cols
print(mat)

# Initialize the random number generator
rng = np.random.default_rng()

# Generate random floats
rand_floats = rng.random(size=(3, 3))
print(rand_floats)
print(rand_floats.shape)
print(rand_floats.dtype)
print(rand_floats[1, 2])