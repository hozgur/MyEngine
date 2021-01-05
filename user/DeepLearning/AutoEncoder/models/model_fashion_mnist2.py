
class Model(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3)
        self.conv2 = nn.Conv2d(16,4,3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.hidden_layer = nn.Linear(4*25,196)
        self.conv1_ = nn.Conv2d(1,64,3,1,1)
        self.conv2_ = nn.Conv2d(64,16,3,1,1)
        self.conv3_ = nn.Conv2d(16,1,3,1,1)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')        
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.batch_size = kwargs["batch_size"]
        self.dataset = torchvision.datasets.FashionMNIST(root=MyEngine.Path("user/DeepLearning/data"), train=True, transform=self.transform, download=True)
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.it = iter(self.loader)                

    def forward(self, features):
        code = self.conv1(features)
        code = torch.relu(code) 
        code = self.maxpool1(code)
        code = self.conv2(code)
        code = torch.relu(code) 
        code= self.maxpool2(code)
        code = code.view(-1,4*25)
        code = self.hidden_layer(code)
        code = torch.relu(code) 
        code = code.view(self.batch_size,1,14,14)
        #code = self.up1(code)
        code = self.conv1_(code)
        code = torch.relu(code) 
        code = self.up2(code) 
        code = self.conv2_(code)
        code = torch.relu(code)
        code = self.conv3_(code)
        code = torch.relu(code)
        code = torch.clamp(code,0,1)
        return code

    def train_(self):
        try:
            self.inp = next(self.it)[0].to(device)
            self.optimizer.zero_grad()    
            self.outp = self.forward(self.inp)
            train_loss = self.criterion(self.outp, self.inp)
            train_loss.backward()
            self.optimizer.step()
        except StopIteration:
            self.it = iter(self.loader)

    def name_(self):
        return "FashionMnist2.pth"