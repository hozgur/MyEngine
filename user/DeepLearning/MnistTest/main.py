import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mytensor
dataset1 = datasets.MNIST('../data', train=True, download=True,transform=None)

tensor = mytensor.MyTensor()
#id = tensor.fromBuffer(dataset1[0][0].tobytes(),(1,28,28))

#print("id = ", id)