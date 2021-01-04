
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=256
        )
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=256, out_features=16
        )       
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=16, out_features=256
        )
        self.decoder_hidden_layer3 = nn.Linear(
            in_features=256, out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )
        self.dp = nn.Dropout(0.1, inplace = True)
        

    def forward(self, features):
        features = features.view(-1,784);
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)        
        code = torch.relu(code)        
        code = self.decoder_hidden_layer1(code)
        code = torch.relu(code)        
        code = self.decoder_hidden_layer2(code)
        code = torch.relu(code)        
        code = self.decoder_hidden_layer3(code)
        code = torch.relu(code)
        activation = self.decoder_output_layer(code)
        reconstructed = torch.relu(activation)
        reconstructed = torch.clamp(reconstructed,0,1)
        return reconstructed.view(-1,1,28,28)

    def forward2(self,x,y):
        inp = torch.Tensor([x,y])        
        code = self.decoder_hidden_layer2(inp)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer3(code)
        activation = torch.relu(activation)                
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
