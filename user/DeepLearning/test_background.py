import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import mytensor
import MyEngine
import time
print(torch.__version__)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

c = 0
tensor = MyEngine.GetBackground()
back = np.asarray(tensor,dtype="byte")
print(back.shape)

im = Image.open(MyEngine.Path("asset/ball64.png"))
nim = np.array(im)
print(nim.shape)

conv1 = nn.Conv2d(1, 1, 3, 1,1)
w = torch.ones(1,1,3,3)/10                                                                              
b = torch.zeros(1)
sw = back.shape[1]
sh = back.shape[0]
#w[0,0,1,1] = 1.5
conv1.weight = torch.nn.parameter.Parameter(w)
conv1.bias = torch.nn.parameter.Parameter(b)
print(conv1.parameters)
def runBatch():    
    
    backt = torch.tensor(back[0:sh,0:sw,0:3],dtype = torch.uint8)
    backt = torch.transpose(backt,0,2).unsqueeze(1)
    out = conv1(backt.float())
    out = out.squeeze().transpose(0,2)
    out2 = out.detach()
    back[0:sh,0:sw,0:3] = out2
     
    #if mouseX > 32 and mouseY > 32:
    #back[mouseY-32:mouseY+32,mouseX-32:mouseX+32] = nim
    return 1
    
