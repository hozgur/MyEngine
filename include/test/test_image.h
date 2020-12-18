#pragma once

#include "my.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	handle hImage = invalidHandle;
	int iW = 48;
	int iH = 48;
	MyEngine(const char* path) :My::Engine(path)
	{
		
	}

	bool OnStart() override
	{
		std::string path = myfs::path("asset/balls48-144.png");
		hImage = loadImage(path);
		AddWindow(400, 300,3,3);
		return true;
	}

	void OnDraw() override
	{
		if (hImage >= 0)
		{
			if (mousePressed)
			{
				int x = std::rand() &3;
				int y = std::rand() & 3;

				//FillCircle({ (int)mouseX, (int)mouseY }, 30, Color::Red);
				DrawImage(hImage, (int)mouseX, (int)mouseY, iW, iH, x* iW, y * iW);
				
			}
			
		}
	}
};