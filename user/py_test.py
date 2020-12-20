import pymyarray
import spam

import numpy as np
arr = pymyarray.PyMyArray(size=10, type="int")
print (arr)

arr2 = spam.system((2,3,4),"double")
nparr = np.asarray(arr2, dtype=np.float32)
print(nparr.shape)
print(nparr.dtype)
nparr[0:5] = 0
print(nparr)

def test_numpy():	
	print("Hello from Numpy function");
	a = np.ones((2,3), dtype=int)
	print(a)
	return a


def test_function(a):
	print(a)
	print("Hello from function");
	return 42


test_function(35)

	
