
clientWidth = 1200
clientHeight = 800
pixelWidth = 1
pixelHeight = 1
datWidth = 300
datHeight = 150
datX = 300
datY = 0

lineCount = 100
--line = {x = 0, y = 0, dx = 0, dy = 0, color = 0}
lines = {}
--engine.addwindow(clientWidth, clientHeight, pixelWidth, pixelHeight, false)

print(g_Var)

for a=1,lineCount do	
	lines[a] = {x = int(clientWidth * math.random()), y = int(clientHeight * math.random()), dx = 3 * math.random(), dy = 2 * math.random(), color = color.random()}	
end

function draw1()
	for a = 1,lineCount/2 do
		
		engine.fillcircle(int(lines[a].x),int(lines[a].y),7,lines[a].color)
		lines[a].x = lines[a].x + lines[a].dx
		lines[a].y = lines[a].y + lines[a].dy
		if(lines[a].x < 0 or lines[a].x >= clientWidth) then
			lines[a].x = lines[a].x - lines[a].dx
			lines[a].dx = - lines[a].dx		
		end

		if(lines[a].y < 0 or lines[a].y >= clientHeight) then
			lines[a].y = lines[a].y - lines[a].dy
			lines[a].dy = - lines[a].dy				
		end
	end
end

function draw2()
	for a = lineCount/2+1,lineCount do
		
		engine.fillcircle(int(lines[a].x),int(lines[a].y),3,lines[a].color)
		lines[a].x = lines[a].x + lines[a].dx
		lines[a].y = lines[a].y + lines[a].dy
		if(lines[a].x < 0 or lines[a].x >= clientWidth) then
			lines[a].x = lines[a].x - lines[a].dx
			lines[a].dx = - lines[a].dx		
		end

		if(lines[a].y < 0 or lines[a].y >= clientHeight) then
			lines[a].y = lines[a].y - lines[a].dy
			lines[a].dy = - lines[a].dy				
		end
	end
end


function move()
	for a = 1,lineCount do
		lines[a].x = lines[a].x + lines[a].dx
		lines[a].y = lines[a].y + lines[a].dy
		if(lines[a].x < 0 or lines[a].x >= clientWidth) then
			lines[a].x = lines[a].x - lines[a].dx
			lines[a].dx = - lines[a].dx		
		end

		if(lines[a].y < 0 or lines[a].y >= clientHeight) then
			lines[a].y = lines[a].y - lines[a].dy
			lines[a].dy = - lines[a].dy				
		end
	end
end


local time = 0

c = color.random();
function OnDraw()
	engine.clear(color.black)
	w = int(70 - 50 * math.abs(math.sin(time/12)))

	time = time + 1
	draw1()	
	move()
	x,y = engine.getmouse();	
	engine.fillrect(int(x-w/2),int(y-w/2),w,w,c)
	engine.drawrect(int(x-w/2),int(y-w/2),w,w,color.white)	
	draw2()
	return 1;
end

function OnKey(key,state)
	--print(key)
	c = color.random();
	return 1;
end

