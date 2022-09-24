
backBuffer = np.asarray(my.GetBackground(),dtype="byte")
import colorsys



def get_color(x,y):	
	return [backBuffer[x,y,0]/255.0,backBuffer[x,y,1]/255.0,backBuffer[x,y,2]/255.0]
	

def set_color(x,y,c):
	backBuffer[x,y,:3] = c


for i in range(0,100):
	for j in range(0,100):
		set_color(i,j,[i,j,0])