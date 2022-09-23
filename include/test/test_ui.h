#pragma once
#include "my.h"

class MyEngine : public My::myEngine
{
public:
	My::Slider *slider;
	MyEngine() :My::myEngine()
	{
		
	}
	
	bool OnStart() override
	{
		AddWindow(600, 600);
		SetFPS(60);
		slider = new My::Slider(clientWidth - 210, 10, 200, 30);
		return true;
	}

	void OnDraw() override
	{
		slider->Draw();
	}

	bool OnUpdate() override
	{
		return false;
	}

	void OnExit() override
	{

	}
};
