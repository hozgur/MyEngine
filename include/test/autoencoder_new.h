#pragma once
#include "my.h"
#include "mypy.h"
#include "mytensor.h"
#include "myparser.h"
class MyEngine : public myEngine
{
public:
	MyEngine(const char* path) :myEngine(path)
	{

	}
	const int width = 800;
	const int height = 600;
	const int posX = 400;
	const int posY = 0;
	const myString project_path = "user/DeepLearning/AutoEncoder2";
	bool status = false;
	myHandle menu = invalidHandle;
	bool OnStart() override
	{
		if (!myPy::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(width, height);
		menu = AddWebView(posX, posY, 400, 300);
		
		/*if (myPy::dofile(myfs::path(project_path,"init.py")))
		{
			status = true;
		}
		else
		{
			debug << "Error on init.py";
			exit(1);
		}*/
		return true;
	}

	void OnReady(myHandle id) override
	{
		myString inpath = myfs::path(project_path, "ui/ui.html");
		myString outpath = myfs::path(project_path, "ui/compiled/ui.html");
		myString jspath = myfs::path(project_path, "ui/ui.js");
		
		myParser::parse(inpath, outpath, {
			{"JS_PATH",jspath}
			});
		Navigate(menu, "file://" + outpath);
		debug << "Ready\n";
	}
	bool OnNavigate(myHandle id, std::string uri) override {
		debug << "Navigate to " << uri << "\n";
		return true;
	}
	void OnKey(uint8_t key, bool pressed) override
	{
		if (pressed)
		{
			if (key == myKey::S) status = myPy::dofunction("save_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
			if (key == myKey::L) status = myPy::dofunction("load_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
		}


	}

	void run()
	{
		status = myPy::dofunction("runBatch", {}) >= 0;		
	}

	void OnIdle() override
	{
		if (status)
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
