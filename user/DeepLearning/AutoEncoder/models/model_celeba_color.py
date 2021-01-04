
class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,4,3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.hidden_layer = nn.Linear(4*25,121)
        self.dconv1 = nn.ConvTranspose2d(1, 8, 3, stride=1)
        self.dconv2 = nn.ConvTranspose2d(8, 1, 4, stride=2)        
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.batch_size = kwargs["batch_size"]
        self.dataset = torchvision.datasets.MNIST(root=MyEngine.Path("user/DeepLearning/data"), train=True, transform=self.transform, download=True)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.it = iter(self.loader)                

    def forward(self, features):
        code = self.conv1(features)
        code = torch.relu(code) 
        code = self.maxpool1(code)
        code = self.conv2(code)
        code = torch.relu(code) 
        code,indices = self.maxpool2(code)
        code = code.view(-1,4*25)
        code = self.hidden_layer(code)
        code = torch.relu(code) 
        code = code.view(self.batch_size,1,11,11)
        code = self.dconv1(code)
        code = torch.relu(code) 
        code = self.dconv2(code)
        code = torch.relu(code)
        code = torch.clamp(code,0,1)
        return code

    def train(self):
        try:
            self.inp = next(self.it)[0].to(device)
            self.optimizer.zero_grad()    
            self.outp = self.forward(self.inp)
            train_loss = self.criterion(self.outp, self.inp)
            train_loss.backward()
            self.optimizer.step()
        except StopIteration:
            self.it = iter(self.loader)
