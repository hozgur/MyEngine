my.Import("user/DeepLearning/autoencoder2/autoencoders.py")
net = CNN()
my.Import("user/python/graphics/encoder.py")
checkPointPath = my.Path("user/DeepLearning/autoencoder2/data/checkpoint.pth.tar")
checkpoint = torch.load(checkPointPath)
net.load_state_dict(checkpoint['state_dict'])
net.to(device)
backBuffer = np.asarray(my.GetBackground(),dtype="byte")
import colorsys

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
	x1 = x - 128
	y1 = y - 128
	inpim = backBuffer[y1:y1+256,x1:x1+256,:3]
	backBuffer[0:256,256:512,0:3] = inpim
	inp = torch.Tensor([inpim.transpose(2,1,0)]).to(device)/255
	permute = [2, 1, 0]
	inp = inp[:,permute]	
	out = encoder(inp)
	print(out)
	outimage = decoder(out)[:,permute]
	
	img_array = sigmoid(outimage).cpu().detach().numpy()	
	img_array = img_array.squeeze()
	img_array = img_array.transpose(1,2,0)
	image = (img_array * 255).astype(np.uint8)	
	#backBuffer[y1:y1+image.shape[1],x1:x1+image.shape[0],:3] = image
	backBuffer[0:0+image.shape[1],0:0+image.shape[0],:3] = image
	
	

onMouseMove(300,300)
onMouseMove(400,300)