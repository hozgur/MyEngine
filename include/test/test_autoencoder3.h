#pragma once
#include "my.h"
#include "mypy.h"
#include "mytensor.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	MyEngine(const char* path) :My::Engine(path)
	{

	}
	
	bool canRun = false;
	bool OnStart() override
	{
		if (!Py::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(800, 500,2,2);
		if (Py::dofile(myfs::path("user/DeepLearning/AutoEncoder/test_autoencoder.py")))
		{			
			canRun = true;
		}
		return true;
	}

	void OnKey(uint8_t key, bool pressed) override
	{
	}

	void run()
	{
		int stat = Py::dofunction("runBatch", {});
		if (stat < 0) { canRun = false; return; }
	}


	void OnIdle() override
	{
		if (canRun)
			run();
	}

	void OnDraw() override
	{
		//Py::dofunction("Forward2", {(int)mouseX,(int)mouseY});
	}

	void OnUpdate() override
	{

	}
};
