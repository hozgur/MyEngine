import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mytensor

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()

print ("datasets loading..")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
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
it = iter(train_loader)
tensor = mytensor.MyTensor()
def runBatch():    
    inp = next(it)[0].view(-1,784)
    optimizer.zero_grad()        
    # compute reconstructions    
    outputs = model(inp)
    out = inp.numpy()        
    id = tensor.fromBuffer(out.tobytes(),(128,28,28))
    print(id)    


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