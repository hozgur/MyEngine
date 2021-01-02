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
torch.cuda.empty_cache()

#  use gpu if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

c = 0
tensor = MyEngine.GetBackground()
back = np.asarray(tensor,dtype="byte")
print(back.shape)

im = Image.open(MyEngine.Path("asset/ball64.png"))
nim = np.array(im)
print(nim.shape)

conv1 = nn.Conv2d(4, 4, 3, 1,1)
w = torch.ones(4,4,3,3)/ 90 -( 1/180)
b = torch.zeros(4)
conv1.weight = torch.nn.parameter.Parameter(w)
conv1.bias = torch.nn.parameter.Parameter(b)
def runBatch():    
    
    backt = torch.tensor(back[0:500,0:1000,0:4])
    backt = torch.transpose(backt,0,2).unsqueeze(0)
    out = conv1(backt.float()/255.0)
    out = out.squeeze().transpose(0,2)
    #print(out.shape)
    back[0:500,0:1000,0:4] = out.detach()*255
    
    #if mouseX > 32 and mouseY > 32:
    back[mouseY-32:mouseY+32,mouseX-32:mouseX+32] = nim
    return 1
    
