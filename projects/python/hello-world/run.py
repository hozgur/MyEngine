
print("Hello World! #2")

def OnReady(id):
	my.Navigate(web1,"www.google.com")

myAnchorNone	= 0
myAnchorLeft	= 1
myAnchorRight	= 2
myAnchorTop		= 4
myAnchorBottom	= 8
myAnchorAll		= 15

my.RemoveWebView(web1)
web1 = my.AddWebView(200,0,400,400,myAnchorRight)




