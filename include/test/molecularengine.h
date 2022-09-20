#pragma once

#include "my.h"
using namespace My;
enum Type { Simple, Wall };
struct dot
{
	dot(vec2 pos, vec2 vel, myColor color, int type = Simple)
	{
		this->pos = pos;
		this->vel = vel;
		this->color = color;
		this->type = type;
	}
	vec2 pos;
	vec2 vel;
	myColor color;
	int type;
};
   
class MyEngine : public My::myEngine
{
public:
	
	std::vector<dot> dots;
	const int dotCountX = 12;
	const int dotCountY = 12;
	const int gap = 5;	
	MyEngine():My::myEngine(1,1)
	{}
	void InitDots()
	{
		int x1 = 0;
		dots.push_back(dot({ 0,0 }, { 0,0 }, myColor::White));
		float y1 = 0.8660254f * gap;
		for (int y = 0; y < dotCountY; y++)
		{
			x1 = x1 + gap / 2;
			for (int x = 0; x < dotCountX; x++)
			{
				dots.push_back(dot({ 50 + x * gap + x1 ,10 + y * y1 }, { 0,0 }, myColor::Random()));
			}
			if (x1 >= gap) x1 = 0;
		}
		for (int x = 0; x < 300; x += 3)
		{
			dots.push_back(dot({ 50 + x ,330 + x * 0 }, { 0,0 }, myColor::Blue, Wall));
		}
	}
	bool OnStart() override
	{
		AddWindow(600, 600, false);
		SetFPS(60);
		std::srand(0);
		InitDots();
		return true;
	}
	void DrawDot(int nDot)
	{
		auto& d = dots[nDot];
		//GetLinePointer((int)d.pos.y())[(int)d.pos.x()] = d.color;
		FillCircle(d.pos.cast<int>(), gap/2, d.color);
	}
	void MoveDot(int nDot)
	{

	}
	void DrawDots()
	{
		for (int i = 0; i < (int)dots.size(); i++)
		{
			DrawDot(i);
			if (i != 0)
				MoveDot(i);
		}
	}
	void fade()
	{
		ForEachPixel([this](int x, int y, myColor& c) {
			const int fader = 100;
			c = myColor(fader * c.r / 255, fader * c.g / 255, fader * c.b / 255);
			});
	}
	void OnDraw() override
	{		
		fade();
		DrawDots();
	}

	bool OnUpdate() override
	{
		
		return false;
	}

	void OnExit() override
	{
		
	}
};