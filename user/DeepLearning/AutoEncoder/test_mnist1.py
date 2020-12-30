import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import mytensor
import MyEngine
import time

#MyEngine.Import("user/DeepLearning/autoencoder/model.py")



batch_size = 16
width = 28
height = 28
print(torch.__version__)
torch.cuda.empty_cache()

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
model = AE(input_shape=width * height).to(device)
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
it = iter(train_loader)
c = 0
start_time = time.time()
def runBatch():
    global it
    try:        
        inp = next(it)[0].view(-1,width * height).to(device)
        optimizer.zero_grad()    
        outp = model(inp)
        train_loss = criterion(outp, inp)
        train_loss.backward()
        optimizer.step()
        c = c + 1
        if c == 20:
            print("--- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            c = 0
            nout = outp.data.cpu().numpy()
            ninp = inp.cpu().numpy()            
            inpId = tensorIn.fromBuffer(ninp.tobytes(),(batch_size,height,width), "float")
            outId = tensorOut.fromBuffer(nout.tobytes(),(batch_size,height,width), "float")            
        return 0    
    except StopIteration:
        it = iter(train_loader)
        return 0
    
def Forward2(x,y):
    global fwdId
    out = model.forward2(x,y)
    nout = out.data.numpy()
    fwdId = tensorDraw.fromBuffer(nout.tobytes(),(1,height,width), "float")

