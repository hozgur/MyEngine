#pragma once
#include "my.h"
#include "mypy.h"
#include "mytensor.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	bool canRun = false;
	MyEngine(const char* path) :My::Engine(path)
	{

	}
	
	bool OnStart() override
	{
		if (!Py::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		SetScript(myfs::path("user/lua_test.lua"));
		AddWindow(1600, 1000);
		canRun = Py::dofile(myfs::path("user/DeepLearning/test_background.py"));
		
		return true;
	}

	void OnKey(uint8_t key, bool pressed) override
	{
		}

	void run()
	{
		Py::setglobal("mouseX", (int)mouseX);
		Py::setglobal("mouseY", (int)mouseY);
		canRun = Py::dofunction("runBatch", {});
		
		
	}
	void OnDraw() override
	{
		if (luaEnable && lua.checkfunction("OnDraw"))
			if (lua.dofunction("OnDraw") == false)
			{
				luaEnable = false;
				debug << lua.error();
			}
	}

	void OnIdle() override
	{		
		if (canRun)
			run();
	}

};