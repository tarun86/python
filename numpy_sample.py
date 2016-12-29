import numpy as np

a = np.array([0, 1, 2, 3, 4, 5])
print a

# ndim
print a.ndim

# shape
print a.shape

# reshape npArray
b = a.reshape(3, 2)
print b

# b & a have the same reference
b[1][1] = 77
print a
print b

# operations are propogated to all the element in numpy array
print a * 2
print a > 2
c = a.copy()
c[c > 2] = 4
print c
# c.clip(lower_limit, upper_limit)

c = np.array([1, 2, np.NaN, 3, 4])
print np.isnan(c)
c = c[~np.isnan(c)]
print c


# numpy namespace is visible to scipy
# inspect their compatibility
import scipy as sp
print sp.version.full_version
print sp.dot is np.dot