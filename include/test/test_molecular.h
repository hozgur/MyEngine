#pragma once
#include "my.h"
#include "molecular/molecular2.h"
using namespace My;


class MyEngine : public My::Engine
{
public:

	const int width = 400;
	const int height = 300;
	grid<int> g;
	MyEngine(const char* path) :My::Engine(path),g(width,height){}
	void InitDots()
	{
		for (int a = 0; a < 10; a++)
			g.add(a * 3., a * 3., pixel<int>(My::Color::Random()));
	}
	bool OnStart() override
	{
		//AddWindow(width, height, 1, 1);
		InitDots();
		
		return true;
	}
	void MoveDot(int nDot)
	{

	}
	
	void OnDraw() override
	{
	
	}

	void OnUpdate() override
	{
		
	}

	void OnExit() override
	{

	}
};