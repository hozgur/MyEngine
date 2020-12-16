#pragma once

#include "my.h"
#include "molecular.h"
using namespace My;
const int scale = 1;
class MyEngine : public My::Engine
{
public:
	
	const int width = 256;
	const int height = 92;
	Molecular::grid grid;
	MyEngine() :My::Engine(),grid(width,height,scale)
	{}
	void InitDots()
	{		
		for (int y = 0; y < 48; y++)
		{
			for (int x = 0; x < 20; x++)
			{
				grid.setPixel(x+15, 1*y+5, Color::Random());
			}
		}

		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				grid.setPixel(x +125, y + 25, Color::Random(),-3500,0,1);
			}
		}
		/*grid.setPixel(10, 10, Color::Red);
		Molecular::cell &c = grid.getCell(10, 10);
		c.pixels[0].vx = -1000;
		c.pixels[0].vy = -2000;*/
		//grid.setPixel(2, 1, Color::Blue);
		//grid.runCell(10, 10);
		//Molecular::pair p = Molecular::attTable2.func(5000, 5000);
		//debug << p;
	}
	bool OnStart() override
	{
		AddWindow(width*scale, height*scale,5,5);		
		std::srand(0);
		grid.bDrawLines = true;
		InitDots();		
		return true;
	}
	void MoveDot(int nDot)
	{

	}
	void DrawDots()
	{
		if(scale == 1)
			ForEachPixel([this](int x, int y, Color& c) {
				c = grid.getColor(x, y);
				});
		else
			grid.foreachCell([&](int x, int y, Molecular::cell& c) {
				c.draw(x, y, scale, scale);
				});
	}
	void fade()
	{
		ForEachPixel([this](int x, int y, Color& c) {const int fader = 100;
		c = Color(fader * c.r / 255, fader * c.g / 255, fader * c.b / 255);
			});
	}
	void mouseF()
	{
		int mx = (int)mouseX;
		int my = (int)mouseY;
		int cx = mx / scale;
		int cy = my / scale;
		int x = (mx - cx*scale) * MAXVALUE /scale;
		int y = (my - cy*scale) * MAXVALUE / scale;

		Molecular::cell& c = grid.getCell(cx, cy);
		c.bDrawBorder = true;
		c.mouseX = x;
		c.mouseY = y;
	}
	void OnDraw() override
	{		
		fade();
		
		DrawDots();
		
	}

	void OnUpdate() override
	{
		mouseF();
		grid.Run();
		//return true;
		
	}

	void OnExit() override
	{

	}
};