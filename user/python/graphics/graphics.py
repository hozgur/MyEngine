import json
my.Import("user/python/graphics/model1.py")
net = CNN()
#checkPointPath = my.Path("user/python/graphics/data/model5.pth")
checkPointPath = my.Path("user/DeepLearning/autoencoder2/model7.pth")
checkpoint = torch.load(checkPointPath)
net.load_state_dict(torch.load(checkPointPath))
net.eval()
net.to(device)
backBuffer = np.asarray(my.GetBackground(),dtype="byte")
import colorsys

chunkSize = 64
width = 1200
height = 900
chunksX = int(width/chunkSize)
chunksY = int(height/chunkSize)

my.Message(json.dumps({"id": "python", "message":"Hello2"}))


def drawImage():
	des = np.asarray(Image.open(my.Path("user/python/graphics/test.png")))
	print(des.shape)
	backBuffer[0:height,0:width,0:3] = des[0:height,0:width,::-1]

def get_color(x,y):	
	return [backBuffer[x,y,0]/255.0,backBuffer[x,y,1]/255.0,backBuffer[x,y,2]/255.0]
	

def set_color(x,y,c):
	backBuffer[x,y,:3] = c

def predict(x,y):
	inpim = backBuffer[y:y+chunkSize,x:x+chunkSize,:3].astype(np.uint8)
	backBuffer[0:chunkSize,chunkSize:2*chunkSize,0:3] = inpim # for debugging
	inpim = inpim.transpose(2,0,1)/255.0
	inp = torch.Tensor([inpim]).to(device)
	permute = [2, 1, 0]
	inp = inp[:,permute]
	out = net.flatten(inp)
	out = net.encoder(out)			
	out = net.decoder(out)
	out = net.unflatten(out)
	out = out[:,permute]
	img = out.data.cpu().numpy()	
	img = img.squeeze()
	img = img.transpose(1,2,0)*255.0
	backBuffer[y:y+img.shape[1],x:x+img.shape[0],:3] = img
	backBuffer[0:0+img.shape[1],0:0+img.shape[0],:3] = img # for debugging
	
def renderScreen():
	for y in range(chunksY):
		for x in range(chunksX):
			predict(x*chunkSize,y*chunkSize)

def onMouseMove(x,y):		
	x1 = x - int(chunkSize/2)
	y1 = y - int(chunkSize/2)
	my.Message(json.dumps({"id": "mouseMove", "message":json.dumps({"x":x,"y":y})}))
	predict(x1,y1)
	
drawImage()