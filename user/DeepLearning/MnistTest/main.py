import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mytensor
dataset1 = datasets.MNIST('../data', train=True, download=True,transform=None)
import numpy as np
tensor = mytensor.MyTensor()
tensor2 = tensor.fromBuffer(dataset1[0][0].tobytes(),(1,28,28),"byte")
print(tensor2)

#tensor2 = mytensor.MyTensor((3,28,28),"float")

#natest = np.array(tensor2)

#print(natest)
