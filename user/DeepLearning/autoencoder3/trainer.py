
my.Import('user/DeepLearning/autoencoder3/model3.py')
def log(msg):
    my.Message(json.dumps({"id": "python", "message":msg}))


trainImage = Image.open(my.Path("user/python/graphics/test.png"))
log("Training Image loaded.")

# Create the model
model = CNN().to(device)
log("Model created.")
trainImageTensor = torch.tensor(np.array(trainImage)).to(device)
log("Training Image Tensor created.")
# create image chunks
chunkSize = 32

# create the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
log("Optimizer created.")

# create the loss function
criterion = nn.MSELoss()
log("Loss function created.")

# train the model
log("Training started.")
for epoch in range(1, 10):
    for i in range(0, trainImageTensor.shape[0], chunkSize):
        for j in range(0, trainImageTensor.shape[1], chunkSize):
            chunk = trainImageTensor[i:i+chunkSize, j:j+chunkSize]
            chunk = torch.transpose(chunk, 0, 2)/255.0            
            optimizer.zero_grad()
            output = model(chunk)            
            loss = criterion(output, chunk)
            loss.backward()
            optimizer.step()
        log("Height: " + str(i) + " Loss: " + str(loss.item()))
    log("Epoch: " + str(epoch) + " Loss: " + str(loss.item()))



