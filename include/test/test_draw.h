#pragma once
#include "my.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	ivec2 start;
	ivec2 stop;
	MyEngine(const char* path) :My::Engine(path)
	{
		
	}

	void OnMouse(MouseEvent evt, float x, float y) override
	{
		switch (evt)
		{
		case MouseEvent::Mouse_LBPressed: start = { (int)x,(int)y }; break;
		case MouseEvent::Mouse_Move:
		case MouseEvent::Mouse_LBReleased: stop = { (int)x,(int)y }; break;
		}
	}
	bool OnStart() override
	{
		AddWindow(1000, 800);
		return true;
	}

	void OnDraw() override
	{
		Clear(Color::Black);
		if(mousePressed)
			FillRect(start, stop-start, Color::Red);
		DrawLine(start, { (int)mouseX, (int)mouseY }, Color::Blue);
	}
};