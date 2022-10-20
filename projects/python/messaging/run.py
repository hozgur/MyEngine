#messaging run.py file

my.Navigate(web1,htmlFile)	#for refresh web controller

image_path = "images/FRUIT.png"



def onloadImage(message):
	print("onLoadImage",message)

def onReady(message):
	print("onReady",message)
	my.WebMessage(web1,"path",image_path)

messageHandlers = {
	"load":onloadImage,
	"ready":onReady
}


def OnMessage(webId,jmsg):
	message = json.loads(jmsg)
	messageHandlers[message["id"]](message["message"])
	