class PixelPredictor(nn.Module):
    def __init__(self, inChannels=3, chunkSizeX = 8,chunkSizeY = 8):
        super(PixelPredictor, self).__init__()
        self.chunkSizeX = chunkSizeX
        self.chunkSizeY = chunkSizeY        
        self.transfer = nn.Sigmoid()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=3*chunkSizeX*chunkSizeY, out_features=64*3),
            nn.ReLU(),
			nn.Linear(in_features = 64*3, out_features=8*3),
            nn.ReLU(),
			nn.Linear(in_features = 8*3, out_features=3),
            nn.ReLU(),
        )                
        self.flatten = nn.Flatten()  
                    
    def forward(self, x):	    
        out = self.flatten(x)
        out = self.encoder(out)
        return out
