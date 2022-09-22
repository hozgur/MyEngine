from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels=3, ch=[16,32,64,128,256]):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=ch[0], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch[0], out_channels=ch[1], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch[1], out_channels=ch[2], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch[2], out_channels=ch[3], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=ch[3], out_channels=ch[4], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            
        )
        self.flatten    = nn.Flatten()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=5*5*ch[4], out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=5*5*ch[4]),
            nn.ReLU()
        )
        self.unflatten  = nn.Unflatten(dim=1, unflattened_size=(ch[4],5,5))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ch[4], out_channels=ch[3], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch[3], out_channels=ch[2], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch[2], out_channels=ch[1], kernel_size=(5,5), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch[1], out_channels=ch[0], kernel_size=(5,5), stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=ch[0], out_channels=in_channels, kernel_size=(5,5), stride=2, output_padding=1),
            
        )
    
    def forward(self, x):
        encoder    = self.encoder(x)
        flatten    = self.flatten(encoder)
        bottleneck = self.bottleneck(flatten)
        unflatten  = self.unflatten(bottleneck)
        decoder    = self.decoder(unflatten)
        sigmoid    = nn.Sigmoid()
        output     = sigmoid(decoder)

        return output

