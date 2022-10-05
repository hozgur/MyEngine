

my.Import("projects/python/hello-world/anchors.py")

print("Hello World! #2")


htmlFile = my.Path("projects/python/hello-world/html/hello.html")

def OnReady(id):
	my.Navigate(web1,htmlFile)


try : web1
except NameError : web1 = 0

	
my.RemoveWebView(web1)
web1 = my.AddWebView(400,0,400,400,myAnchorRight)




