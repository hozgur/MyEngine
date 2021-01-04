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

batch_size = 20

MyEngine.Import(MyEngine.Path("user/DeepLearning/AutoEncoder/model.py"))
class AE2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(1,1,3)
        self.conv2 = nn.Conv2d(1,1,3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.hidden_layer = nn.Linear(25,25)
        self.dconv1 = nn.ConvTranspose2d(1, 1, 3, stride=1)
        self.dconv2 = nn.ConvTranspose2d(1, 1, 4, stride=2)
        self.dconv3 = nn.ConvTranspose2d(1, 1, 3, stride=1)
        self.unmaxpool1 = nn.MaxUnpool2d(2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool2d(2,stride=2)
        
    def forward(self, features):
        code = self.conv1(features)
        code = torch.relu(code) 
        code = self.maxpool1(code)
        code = self.conv2(code)
        code = torch.relu(code) 
        code,indices = self.maxpool2(code)
        code = code.view(-1,25)
        code = self.hidden_layer(code)
        code = torch.relu(code) 
        code = code.view(batch_size,1,5,5)
        code = self.unmaxpool1(code,indices,output_size=torch.Size([batch_size, 1, 11, 11]))
        code = self.dconv1(code)
        code = torch.relu(code) 
        code = self.dconv2(code)
        code = torch.relu(code)
        code = torch.clamp(code,0,1)
        return code

class AE3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,4,3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.hidden_layer = nn.Linear(4*25,121)
        self.dconv1 = nn.ConvTranspose2d(1, 8, 3, stride=1)
        self.dconv2 = nn.ConvTranspose2d(8, 1, 4, stride=2)
        self.unmaxpool1 = nn.MaxUnpool2d(2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool2d(2,stride=2)
        
    def forward(self, features):
        code = self.conv1(features)
        code = torch.relu(code) 
        code = self.maxpool1(code)
        code = self.conv2(code)
        code = torch.relu(code) 
        code,indices = self.maxpool2(code)
        code = code.view(-1,4*25)
        code = self.hidden_layer(code)
        code = torch.relu(code) 
        code = code.view(batch_size,1,11,11)
        #code = self.unmaxpool1(code,indices,output_size=torch.Size([batch_size, 1, 11, 11]))
        code = self.dconv1(code)
        code = torch.relu(code) 
        code = self.dconv2(code)
        code = torch.relu(code)
        code = torch.clamp(code,0,1)
        return code


width = 28
height = 28
print(torch.__version__)
torch.cuda.empty_cache()

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
model = AE3(input_shape=width * height).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print ("datasets loading..")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root=MyEngine.Path("user/DeepLearning/data"), train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)


print ("epochs started..")
epochs = 1
counter = 0
c = 0
back = np.asarray(MyEngine.GetBackground(),dtype="byte")
it = iter(train_loader)

#back[0:28,0:28,0:4] = inp[0:1,0:28,0:28]*255
start_time = time.time()


def printSample(sample,x,y):
    count = sample.shape[0]
    w = sample.shape[3]
    h = sample.shape[2]    
    inp = torch.transpose(sample,1,3)
    inp = torch.transpose(inp,1,2).expand(-1, -1,-1, 4).cpu().detach()
    for i in range(count):
        back[y:y+h,x+i*w:x+i*w+w,0:4] = inp[i:i+1,0:h,0:w]*255


def runBatch():
    global it,c,start_time
    try:        
        inp = next(it)[0].to(device)
        optimizer.zero_grad()    
        outp = model(inp)
        train_loss = criterion(outp, inp)
        train_loss.backward()
        optimizer.step()
        
        c = c + 1
        if c == 120:
            c = 0
            printSample(inp,0,0)
            printSample(outp,0,50)
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()           
        return 0    
    except StopIteration:
        it = iter(train_loader)
        return 0
    
def Forward2(x,y):
    inp = next(it)[0]
    printSample(inp,x,y)

