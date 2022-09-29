import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, inChannels=3, nF=[16,32,64,128,256]):
        super(CNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*32*32, out_features=32*32),
            nn.ReLU(),
            nn.Linear(in_features=32*32, out_features=16*16),
            nn.ReLU(),
            nn.Linear(in_features=16*16, out_features=8*8),
            nn.ReLU(),                        
        )                
                
        self.decoder = nn.Sequential(
            nn.Linear(in_features=8*8, out_features=16*16),
            nn.ReLU(),
            nn.Linear(in_features=16*16, out_features=32*32),
            nn.ReLU(),
            nn.Linear(in_features=32*32, out_features=3*32*32),
            nn.ReLU(),                        
        )
    
    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out