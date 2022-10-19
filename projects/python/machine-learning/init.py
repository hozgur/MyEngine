#machine-learning init.py file
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import datasets, transforms
import mytensor
import MyEngine as my
import time
from PIL import Image
my.Import(project_folder+ "anchors.py")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
windowWidth = 800
windowHeight = 600
if not my.AddMainWindow("Machine Learning",windowWidth,windowHeight):
	print("error on window creation")

try : web1
except NameError : web1 = 0	
my.RemoveWebView(web1)
web1 = my.AddWebView(0,0,windowWidth,200,myAnchorRight)
