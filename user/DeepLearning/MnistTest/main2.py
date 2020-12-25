import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mytensor
gcode = None
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=128, out_features=2
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=2, out_features=128
        )
        self.decoder_hidden_layer3 = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )
        self.dp = nn.Dropout(0.1, inplace = True)
        

    def forward(self, features):
        global gcode
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)        
        code = torch.relu(code)        
        code = self.decoder_hidden_layer1(code)
        code = torch.relu(code)
        gcode = code.data.numpy()
        code = self.decoder_hidden_layer2(code)
        code = torch.relu(code)        
        activation = self.decoder_hidden_layer3(code)
        activation = torch.relu(activation)                
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    def forward2(self,x,y):
        inp = torch.Tensor([x,y])        
        code = self.decoder_hidden_layer2(inp)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer3(code)
        activation = torch.relu(activation)                
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

    #  use gpu if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(input_shape=784)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# mean-squared error loss
criterion = nn.MSELoss()

print ("datasets loading..")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=0
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=0
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
def runBatch():
    global inpId, outId,it,codeId
    try:        
        inp = next(it)[0].view(-1,784)
        optimizer.zero_grad()        
        # compute reconstructions    
        outp = model(inp)
        train_loss = criterion(outp, inp)
        train_loss.backward()
        optimizer.step()
        nout = outp.data.numpy()
        ninp = inp.numpy()            
        inpId = tensorIn.fromBuffer(ninp.tobytes(),(128,28,28), "float")
        outId = tensorOut.fromBuffer(nout.tobytes(),(128,28,28), "float")        
        return 0    
    except StopIteration:
        it = iter(train_loader)
        return 0
    
def Forward2(x,y):
    global fwdId
    out = model.forward2(x,y)
    nout = out.data.numpy()
    fwdId = tensorDraw.fromBuffer(nout.tobytes(),(1,28,28), "float")



#for epoch in range(epochs):    
#    loss = 0    
#    for batch_features, _ in train_loader:                
#        # reshape mini-batch data to [N, 784] matrix
#        # load it to the active device
#        batch_features = batch_features.view(-1, 784)
#        
#        # reset the gradients back to zero
#        # PyTorch accumulates gradients on subsequent backward passes
#        optimizer.zero_grad()
#        
#        # compute reconstructions
#        outputs = model(batch_features)
#        out = batch_features.numpy()
#        tensor = mytensor.MyTensor()
#        id = tensor.fromBuffer(out.tobytes(),(128,28,28))
#        print(out)
#
#        # compute training reconstruction loss
#        train_loss = criterion(outputs, batch_features)
#        
#        # compute accumulated gradients
#        train_loss.backward()
#        
#        # perform parameter update based on current gradients
#        optimizer.step()
#        
#        # add the mini-batch training loss to epoch loss
#        loss += train_loss.item()  
#        if counter == 50 :
#            print("step")
#            counter = 0
#    # compute the epoch training loss
#    loss = loss / len(train_loader)    
#    # display the epoch training loss
#    print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))