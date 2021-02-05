#pragma once
#include "my.h"
#include "molecular/molecular2.h"

using namespace My;

class MyEngine : public My::Engine
{
public:
	typedef int testtype;
	const int width = 1400;
	const int height = 900;
	std::vector<pixel<testtype>> pixels;
	MyEngine(const char* path) :My::Engine(path) {}

	float frand() { return (float)std::rand() / RAND_MAX; }
	void InitDots()
	{
		int size = (width / 2) * (height / 2);
		debug << "Dot Count = " << size << "\n";
		pixels.reserve(size);
		for (int y = 0; y < height; y+=2)
		{
			for (int x = 0; x < width; x+=2)
			{
				pixels.push_back(pixel<testtype>(x, y, Color::Random(),(testtype) frand(), (testtype) frand()));
			}
		}
	}
	bool OnStart() override
	{
		AddWindow(width, height, 1, 1);
		InitDots();
		StopWatch s;
		s.Start();
		const int testCount = 1000;
		for (int a = 0; a < testCount; a++)
		{
			draw();
		}
		s.Stop();
		debug << s.GetDurationStr();
		return true;
	}
	void MoveDot(int nDot)
	{

	}

	void draw()
	{
		for (pixel<testtype>& p : pixels)
		{
			int x = (int)p.x;
			int y = (int)p.y;
			Pixel(p.x, p.y) = p.color;
		}
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