﻿import MyEngine
import mytensor
import numpy as np

tensor = mytensor.MyTensor((2,2),"int")
print(tensor)

nptensor = np.asarray(tensor)
print(nptensor.shape)
print(nptensor.dtype)

arr2 = MyEngine.engine_test((2,3,4),"double")
nparr = np.asarray(tensor, dtype=np.float32)
print(nparr.shape)
print(nparr.dtype)
nparr[0:5] = 0
# print(nparr)

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

	
