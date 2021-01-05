import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, transforms
import mytensor
import MyEngine
import time

MyEngine.Import(MyEngine.Path("user/DeepLearning/AutoEncoder/models/model_fashion_mnist.py"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
model = Model(batch_size=16).to(device)
back = np.asarray(MyEngine.GetBackground(),dtype="byte")

def printSample(sample,x,y):
    count = sample.shape[0]
    w = sample.shape[3]
    h = sample.shape[2]    
    inp = torch.transpose(sample,1,3)
    inp = torch.transpose(inp,1,2).expand(-1, -1,-1, 4).cpu().detach()
    for i in range(count):
        back[y:y+h,x+i*w:x+i*w+w,0:4] = inp[i:i+1,0:h,0:w]*255

c = 0
d = 0
line = 1
sline = 2
start_time = time.time()
def runBatch():
    global c,d,start_time,line,sline
    model.train_()
    c = c + 1
    if c > 50:
        c = 0
        d = d + 1
        if d > 10:
            d = 0
            line = line + 1
            if line > 15:
                line = sline
                sline = sline + 1
                if sline > 15:
                    sline = 15
        printSample(model.inp,0,0)
        printSample(model.outp,0,28 * line)
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time() 
    return 0
    
def save(path):
        torch.save(model.state_dict(), path+model.name_())
        #torch.save(model,path+model.name())
        print("model saved.")

def load(path):
    global model
    #model = torch.load(path+model.name());    
    model = Model(batch_size=20)
    model.state_dict(torch.load(path+model.name_()))
    model.eval()
    print("model loaded.")
        
