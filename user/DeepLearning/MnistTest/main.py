import mytensor
from torchvision import datasets
dataset1 = datasets.MNIST('../data', train=True, download=True,transform=None)
import numpy as np
tensor = mytensor.MyTensor((1,28,28),"float")


def doBatch(c):
    #id = tensor.fromBuffer(dataset1[c][0].tobytes(),(1,28,28),"byte")
    return c

def doBatch2(c):
    id = tensor.fromBuffer(dataset1[c][0].tobytes(),(1,28,28),"byte")
    return id

#tensor2 = tensor.fromBuffer(dataset1[0][0].tobytes(),(1,28,28),"byte")
print(tensor)

#tensor2 = mytensor.MyTensor((3,28,28),"float")

#natest = np.array(tensor2)

#print(natest)