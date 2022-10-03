
print("Hello World! #2")
import MyEngine as my
if not my.AddWindow("Hello World Window",800,600):
	print("error on window creation")

myAnchorNone	= 0
myAnchorLeft	= 1
myAnchorRight	= 2
myAnchorTop		= 4
myAnchorBottom	= 8
myAnchorAll		= 15

def OnReady(id):
	my.Navigate(web1,"www.google.com")


web1 = my.AddWebView(0,0,100,100,myAnchorRight)



