#pragma once
#include "my.h"

using namespace My;
class MyEngine : public My::myEngine
{
public:
	ivec2 start;
	ivec2 stop;
	MyEngine(const char* path) :My::myEngine(path)
	{
		
	}

	void OnMouse(myMouseEvent evt, float x, float y) override
	{
		switch (evt)
		{
		case myMouseEvent::Mouse_LBPressed: start = { (int)x,(int)y }; break;
		case myMouseEvent::Mouse_Move:
		case myMouseEvent::Mouse_LBReleased: stop = { (int)x,(int)y }; break;
		}
	}
	bool OnStart() override
	{
		AddWindow(1000, 800);
		return true;
	}

	void OnDraw() override
	{
		Clear(myColor::Black);
		if(mousePressed)
			FillRect(start, stop-start, myColor::Red);
		DrawLine(start, { (int)mouseX, (int)mouseY }, myColor::Blue);
	}
};