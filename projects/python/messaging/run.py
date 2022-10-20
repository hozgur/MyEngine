#messaging run.py file

my.Navigate(web1,htmlFile)	#for refresh web controller

image_path = "images/FRUIT.png"
thumb_path = "images/THUMB.png"

def onloadImage(message):	
	im_path = my.Path(project_path + message)
	image = Image.open(im_path)	
	size = 100,100
	image.thumbnail(size)		
	file_name = project_path + thumb_path
	full_path = my.Path(file_name)	
	image.save(full_path)
	my.WebMessage(web1,"load",thumb_path)

def onReady(message):
	my.WebMessage(web1,"path",image_path)

messageHandlers = {
	"load":onloadImage,
	"ready":onReady
}


def OnMessage(webId,jmsg):
	message = json.loads(jmsg)
	messageHandlers[message["id"]](message["message"])
	