import json
my.Import("user/python/graphics/model1.py")
net = CNN()
checkPointPath = my.Path("user/python/graphics/data/model1.pth")
checkpoint = torch.load(checkPointPath)
net.load_state_dict(torch.load(checkPointPath))
net.eval()
net.to(device)
backBuffer = np.asarray(my.GetBackground(),dtype="byte")
import colorsys

chunkSize = 32
my.Message(json.dumps({"id": "python", "message":"Hello2"}))
des = np.asarray(Image.open(my.Path("user/python/graphics/test.png")))
print(des.shape)
backBuffer[0:900,0:1200,0:3] = des[0:900,0:1200,::-1]

def get_color(x,y):	
	return [backBuffer[x,y,0]/255.0,backBuffer[x,y,1]/255.0,backBuffer[x,y,2]/255.0]
	

def set_color(x,y,c):
	backBuffer[x,y,:3] = c


def onMouseMove(x,y):
	global backBuffer
	sigmoid = torch.nn.Sigmoid()
	x1 = x - int(chunkSize/2)
	y1 = y - int(chunkSize/2)
	my.Message(json.dumps({"id": "mouseMove", "message":json.dumps({"x":x,"y":y})}))
	inpim = backBuffer[y1:y1+chunkSize,x1:x1+chunkSize,:3].astype(np.uint8)
	backBuffer[0:chunkSize,chunkSize:2*chunkSize,0:3] = inpim
	print(inpim.max())
	print(inpim.min())
	inpim = np.asarray(inpim)/255.0
	print(type(inpim))
	print(inpim.max())
	print(inpim.min())
	inpim = inpim.transpose(2,0,1)
	inp = torch.Tensor([inpim]).to(device)
	permute = [2, 1, 0]
	inp = inp[:,permute]
	#out = net(inp)
	out = net.flatten(inp)
	out = net.encoder(out)
	#my.Message(json.dumps({"id": "python", "message":str(out)}))
	out = net.decoder(out)
	out = net.unflatten(out)
	out = out[:,permute]
	
	img_array = out.data.cpu().numpy()	
	img_array = img_array.squeeze()
	img_array = img_array.transpose(1,2,0)
	back = backBuffer[y1:y1+chunkSize,x1:x1+chunkSize,:3].astype(np.uint8)/255.0	
	img_array +=0.2
	img_array[img_array > 1] = 1
	cmy = 1 - img_array
	backcmy = 1 - back
	render = backcmy + (1-backcmy)*cmy
	render[render > 1] = 1
	rgbrender = 1 - render
	image = (rgbrender * 255).astype(np.uint8)
	#image2 = Image.fromarray(image)
	#image2.save("./out.png")
	backBuffer[y1:y1+image.shape[1],x1:x1+image.shape[0],:3] = image

	backBuffer[0:0+image.shape[1],0:0+image.shape[0],:3] = image
	
	
	

onMouseMove(128,128)