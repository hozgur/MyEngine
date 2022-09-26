import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, inChannels=3, nF=[16,32,64,128,256]):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=nF[0], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nF[0], out_channels=nF[1], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nF[1], out_channels=nF[2], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nF[2], out_channels=nF[3], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=nF[3], out_channels=nF[4], kernel_size=(4,4), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten    = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=nF[4], out_features=nF[4]),
            nn.ReLU(),            
        )
        self.unflatten  = nn.Unflatten(dim=1, unflattened_size=(nF[4],1,1))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nF[4], out_channels=nF[3], kernel_size=(4,4), stride=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=nF[3], out_channels=nF[2], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=nF[2], out_channels=nF[1], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=nF[1], out_channels=nF[0], kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(in_channels=nF[0], out_channels=inChannels, kernel_size=(3,3), stride=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        encoder    = self.encoder(x)
        flatten    = self.flatten(encoder)
        bottleneck = self.bottleneck(flatten)
        unflatten  = self.unflatten(bottleneck)
        decoder    = self.decoder(unflatten)
        sigmoid    = nn.Sigmoid()
        #output     = sigmoid(decoder)
        output     = decoder
        return output
