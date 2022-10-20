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

project_path = "projects/python/messaging/"
my.Import(project_path+ "anchors.py")
htmlFile = my.Path(project_path + "index.html")


try : web1
except NameError : web1 = 0	
my.RemoveWebView(web1)
web1 = my.AddWebView(0,0,windowWidth,200,myAnchorRight)

def OnReady(id):
	my.Navigate(web1,htmlFile)

