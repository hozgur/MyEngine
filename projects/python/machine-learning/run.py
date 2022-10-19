#machine-learning run.py file
print("running.")
project_folder = "projects/python/machine-learning/"

htmlFile = my.Path(project_folder + "html/index.html")

def OnReady(id):
	my.Navigate(web1,htmlFile)


my.Import(project_folder+"models.py")

model = PixelPredictor()

image_path = "images/FRUIT.png"

my.WebMessage(web1,"path",image_path)