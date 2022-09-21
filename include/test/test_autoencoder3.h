#pragma once
#include "my.h"
#include "mypy.h"
#include "mytensor.h"

class MyEngine : public myEngine
{
public:
	MyEngine(const char* path) :myEngine(path)
	{

	}
	
	bool canRun = false;
	bool OnStart() override
	{
		if (!myPy::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(800, 500,2,2);
		if (myPy::dofile(myfs::path("user/DeepLearning/AutoEncoder/test_autoencoder.py")))
		{			
			canRun = true;
		}
		return true;
	}

	void OnKey(uint8_t key, bool pressed) override
	{
		if (pressed)
		{
			if (key == myKey::S) canRun = myPy::dofunction("save_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
			if (key == myKey::L) canRun = myPy::dofunction("load_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
		}
			

	}

	void run()
	{
		int stat = myPy::dofunction("runBatch", {});
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
