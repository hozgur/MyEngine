#pragma once
#include "my.h"
#include "test_perf_molecular.h"
using namespace My;
class MyEngine : public My::Engine
{
public:

	const int width = 400;
	const int height = 300;
	
	MyEngine(const char* path) :My::Engine(path){}
	void InitDots()
	{
		
	}
	bool OnStart() override
	{
		AddWindow(width, height, 1, 1);
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