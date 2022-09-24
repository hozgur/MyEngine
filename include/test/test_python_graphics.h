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
	const int width = 1200;
	const int height = 900;

	const int menuWidth = 400;
	const int menuHeight = height;
	const int posX = width - menuWidth;
	const int posY = 0;
	const myString project_path = "user/python/graphics";
	bool status = false;
	myHandle menu = invalidHandle;
	myColor brushColor = myColor::White;
	static int brushSize;

	bool OnStart() override
	{
		if (!myPy::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(width, height);
		SetWindowTitle("Graphics Test");
		menu = AddWebView(posX, posY, menuWidth, menuHeight, myAnchorRight);

		if (myPy::dofile(myfs::path(project_path, "init.py")))
		{
			status = true;
			debug << "Python ready.\n";
		}
		else
		{
			debug << "Error on init.py\n";
			exit(1);
		}
		reloadModule();	// first load

		return true;
	}

	void reloadModule() {
		myPy::dofile(myfs::path(project_path, "graphics.py"));
	}

	void OnReady(myHandle id) override
	{
		myString inpath = myfs::path(project_path, "ui/ui.html");
		myString outpath = myfs::path(project_path, "ui/compiled/ui.html");
		myString jspath = myfs::path(project_path, "ui/ui.js");
		myString csspath = myfs::path(project_path, "ui/ui.css");
		myString libpath = myfs::path("script/web/lib/");

		myParser::parse(inpath, outpath, {
			{"LIB_PATH",libpath},
			{"JS_PATH",jspath},
			{"CSS_PATH",csspath}
			});
		Navigate(menu, "file://" + outpath);

	}

	void OnMessageReceived(myHandle id, myString msg) override
	{
		json jmsg = json::parse(msg);

		myString id2 = jmsg["id"];
		myString value = jmsg["message"];
		if (id2 == "reload")
		{
			reloadModule();
		}

		if (id2 == "color")
		{	
			json color = json::parse(value);
			brushColor = myColor(color[0], color[1], color[2]);
			debug << "Color: " << color[0] << "\n";

		}
		if (id2 == "brushSize")
		{		
			json brush = json::parse(value);
			brushSize = brush;
			debug << "Brush Size: " << brushSize << "\n";
		}

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
		//status = myPy::dofunction("runBatch", {}) >= 0;
	}

	void OnIdle() override
	{
		static int oldx = 0, oldy = 0;
		if (mousePressed) {
			if (mouseX != oldx || mouseY != oldy) {
				oldx = mouseX;
				oldy = mouseY;
				myPy::dofunction("onMouseMove", { mouseX,mouseY });
			}
		}
	}
	int clamp(float val,float max = 2000) {
		if (val < 0) return 0;
		if (val > max) return (int) max;
		return (int)val;
	}
	

	void OnDraw() override
	{
		//myEngine::OnDraw();
		//Py::dofunction("Forward2", {(int)mouseX,(int)mouseY});	
		static int oldx, oldy;
		int x = (int)mouseX;
		int y = (int)mouseY;
		if (mousePressed) {
			DrawLine(oldx, oldy, x, y, brushColor, [](int x, int y, myColor p) {
				myEngine::pEngine->FillCircle({ x,y }, brushSize, p);
				});
			
			//myPy::dofunction("onMouseMove", { mouseX,mouseY });
		}
		oldx = x;
		oldy = y;			
	}

		void OnUpdate() override
		{

		}

};

int MyEngine::brushSize = 5;