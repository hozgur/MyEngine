import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mytensor
import MyEngine
import time

MyEngine.Import("user/DeepLearning/autoencoder/model.py")
print ("test1.py running")
gcode = None
batch_size = 1
width = 178
height = 218
print(torch.__version__)
torch.cuda.empty_cache()


    #  use gpu if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=width * height).to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()


print ("datasets loading..")
transform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()
                                    ])
path = MyEngine.Path("user/DeepLearning/data/Celeba");
print(path)
train_dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)
#tiff header invalid.

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
)


print ("epochs started..")
epochs = 1
counter = 0
tensorIn = mytensor.MyTensor()
tensorOut = mytensor.MyTensor()
tensorCode = mytensor.MyTensor()
tensorDraw = mytensor.MyTensor()
it = iter(train_loader)
inpId = 0
outId = 0
codeId = 0
fwdId = 0
c = 0
start_time = time.time()
def runBatch():
    global inpId, outId,it,codeId,c,start_time
    try:        
        inp = next(it)[0].view(width*height).to(device)
        optimizer.zero_grad()        
        # compute reconstructions    
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
