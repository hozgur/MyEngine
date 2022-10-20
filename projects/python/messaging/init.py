#messaging init.py file

import numpy as np
import mytensor
import MyEngine as my
import time
import json
from PIL import Image


windowWidth = 800
windowHeight = 600
if not my.AddMainWindow("Messaging Example",windowWidth,windowHeight):
	print("error on window creation")

project_folder = "projects/python/messaging/"
my.Import(project_folder+ "anchors.py")
htmlFile = my.Path(project_folder + "index.html")


try : web1
except NameError : web1 = 0	
my.RemoveWebView(web1)
web1 = my.AddWebView(0,0,windowWidth,200,myAnchorRight)

def OnReady(id):
	my.Navigate(web1,htmlFile)

