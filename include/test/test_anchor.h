#pragma once
#include "my.h"
#include "mypy.h"

class MyEngine : public myEngine
{
public:
	MyEngine(const char* path) :myEngine(path)
	{

	}
	const int width = 1200;
	const int height = 900;

	const int menuWidth = 400;
	const int menuHeight = height;
	const int posX = width - menuWidth;
	const int posY = 0;
	const myString project_path = "user/DeepLearning/AutoEncoder2";
	bool status = false;
	myHandle menu = invalidHandle;
	myColor brushColor = myColor::White;
	bool OnStart() override
	{
		AddWindow(width, height);
		SetWindowTitle("Anchor Test");
		menu = AddWebView(posX, posY, menuWidth, menuHeight, myAnchorRight | myAnchorBottom | myAnchorTop);
		return true;
	}

	void OnReady(myHandle id) override
	{		
		Navigate(menu, "www.google.com");
	}

	void OnMessageReceived(myHandle id, myString msg) override
	{
		
	}
};
