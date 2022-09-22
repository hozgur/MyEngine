

net = CNN()
checkPointPath = MyEngine.Path("user/DeepLearning/autoencoder2/data/checkpoint.pth.tar")
checkpoint = torch.load(checkPointPath)
net.load_state_dict(checkpoint['state_dict'])
net.to(device)

parameters = {}
for i in range(16):
    parameters["param"+str(i)] = 0

def onChange(id,value):
    parameters[id] = float(value)
    params = np.array(list(parameters.values()))
    print(params)
    params = torch.FloatTensor([params]).to(device)
    decodeImage(params)


decoder = torch.nn.Sequential(
    net.bottleneck[4],
    net.bottleneck[5],
    net.bottleneck[6],
    net.bottleneck[7],
    net.unflatten,
    net.decoder
)

decoder = decoder.to(device)


def decodeImage(params):
        
    sigmoid = torch.nn.Sigmoid()
    output = decoder(params)
    output = sigmoid(output)

    img_array = output.data.cpu().numpy()
    img_array = img_array.squeeze()
    img_array = img_array.transpose(1,2,0)
    image = (img_array * 255).astype(np.uint8)

    backBuffer = np.asarray(MyEngine.GetBackground(),dtype="byte")
    backBuffer[0:0+image.shape[1],0:0+image.shape[0],:3] = image