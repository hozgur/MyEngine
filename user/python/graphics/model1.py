class CNN(nn.Module):
    def __init__(self, inChannels=3, chunkSizeX = 64,chunkSizeY = 64):
        super(CNN, self).__init__()
        self.chunkSizeX = chunkSizeX
        self.chunkSizeY = chunkSizeY
        self.intermediateSize = 1024
        self.transfer = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=3*chunkSizeX*chunkSizeY, out_features=self.intermediateSize),
            nn.ReLU(),
        )                
        self.flatten = nn.Flatten()  
        self.unflatten  = nn.Unflatten(1, (3,chunkSizeY,chunkSizeX))

        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.intermediateSize, out_features=3*chunkSizeX*chunkSizeY),
            self.transfer,
        )
    
    def forward(self, x):
        out = self.flatten(x)
        out = self.encoder(out)
        out = self.decoder(out)
        out = self.unflatten(out)
        return out
