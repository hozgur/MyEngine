print("running")

imagepath = MyEngine.Path("user/DeepLearning/autoencoder2/images/FRUIT-SMALL.PNG")

image = Image.open(imagepath).rotate(angle)
imArray = np.asarray(image,dtype="byte")
imArray2 = imArray[:,:,::-1].copy()
imgTensor = torch.tensor(imArray2)

backBuffer = np.asarray(MyEngine.GetBackground(),dtype="byte")

# copy  image data to back buffer on x,y position


backBuffer[0:0+image.size[1],0:0+image.size[0],:3] = imgTensor

