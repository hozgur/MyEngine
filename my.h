#pragma once
// my.h

// Declaration place
#include "core.h"
#include "myview.h"
#include "myimage.h"
#include "mywebview.h"
#include "platform.h"
#include "commandqueue.h"
#include "engine.h"
#include "mylua.h"

#ifdef USE_MY_CONTROLS
#include "ui\theme.h"
#define currentTheme DefaultTheme
#include "ui\control.h"
#include "ui\slider.h"
#endif


/* 
#include "my.h"
using namespace My;


class MyEngine : public My::Engine
{
public:
	MyEngine() :My::Engine()
	{}

	bool OnStart() override
	{
		AddWindow(600, 600, false);		
		return true;
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

*/

