import pymyarray
import numpy as np
arr = pymyarray.PyMyArray(size=10, type="int")
print (arr)

nparr = np.asarray(arr, dtype=np.float32)
print(nparr.shape)
print(nparr.dtype)
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

	
