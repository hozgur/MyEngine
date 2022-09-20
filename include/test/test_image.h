#pragma once

#include "my.h"

using namespace My;
class MyEngine : public My::myEngine
{
public:
	myHandle hImage = invalidHandle;
	int iW = 64;
	int iH = 64;
	MyEngine(const char* path) :My::myEngine(path)
	{
		
	}

	bool OnStart() override
	{
		std::string path = myfs::path("asset/ball64.png");
		hImage = loadImage(path);
		AddWindow(1400, 900,1,1);
		return true;
	}

	void OnDraw() override
	{
		if (hImage >= 0)
		{
			if (mousePressed)
			{
				/*int x = std::rand() &3;
				int y = std::rand() & 3;*/
				int x = 0,y = 0;

				//FillCircle({ (int)mouseX, (int)mouseY }, 30, myColor::Red);
				DrawImage(hImage, (int)mouseX- iW/2, (int)mouseY - iH/2, iW, iH, x* iW, y * iW);
				
			}
			
		}
	}
};