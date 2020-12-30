import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import mytensor
import MyEngine
import time
from PIL import Image
print(torch.__version__)
torch.cuda.empty_cache()

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)

c = 0
tensor = MyEngine.GetBackground()
ntens = np.asarray(tensor,dtype="byte")
print(ntens.shape)

im = Image.open(MyEngine.Path("asset/ball64.png"))
nim = np.array(im)
print(nim.shape)
def runBatch():    
    ntens[mouseY-32:mouseY+32,mouseX-32:mouseX+32] = nim
    return 0
    
