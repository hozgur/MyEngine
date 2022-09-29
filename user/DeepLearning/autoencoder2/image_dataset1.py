import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class ImageDataset1(Dataset):
    

    def __init__(self, image_path="test-cropped.png", chunkSize = 64,transform=None,device="cuda:0"):
        
        self.transform = transform
        trainImage = Image.open(image_path)
        chunkSizeX = chunkSize
        chunkSizeY = chunkSize
        trainImageTensor = torch.tensor(np.array(trainImage))/255.    
        width = trainImageTensor.shape[1]
        height = trainImageTensor.shape[0]        

        xChunks = int((width)/chunkSize)
        yChunks = int((height)/chunkSize)
        chunks = xChunks * yChunks    
        stride4 = 1
        stride3 = 3
        stride2 = width*3
        stride1 = chunkSize
        stride0 = chunkSize*width*3
        trainImageChunks = trainImageTensor.as_strided((yChunks, xChunks,chunkSize, chunkSize,3), (stride0,stride1, stride2, stride3, stride4))
        
        self.trainImageChunks = trainImageChunks.flatten(0,1)
    
    def __len__(self):
        return len(self.trainImageChunks)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.trainImageChunks[idx].transpose(0,2).transpose(1,2)

        if self.transform:
            sample = self.transform(sample)
            
        return [sample,sample]