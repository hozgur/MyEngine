#pragma once

#include "my.h"
#include "mypy.h"
#include "mytensor.h"
#include "myparser.h"



class MyTrainer : public myEngine
{
public:
	MyTrainer(const char* path) :myEngine(path)
	{

	}
	virtual ~MyTrainer()
	{		
		debug << "Trainer destroyed.";
	}
	
	const int wndWidth = 400;
	const int wndHeight = 800;

	const myString appTitle = "AutoEncoder Trainer";
	const myString project_path = "user/DeepLearning/autoencoder3";
	const myString python_file = "trainer.py";

	bool status = false;
	myHandle menu = invalidHandle;
	myHandle messageBox = invalidHandle;
	

	bool OnStart() override
	{
		if (!myPy::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		AddWindow(wndWidth, wndHeight);
		SetWindowTitle(appTitle);
		menu = AddWebView(0, 0, 400, 200, myAnchorRight);
		messageBox = AddWebView(0, wndHeight - 200, wndWidth, 200, myAnchorLeft | myAnchorRight | myAnchorBottom);
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
		myPy::dofile(myfs::path(project_path, python_file));
	}

	void navigate(myHandle view, myString html, myString js) {
		myString inpath = myfs::path(project_path, "ui/" + html);
		myString outpath = myfs::path(project_path, "ui/compiled/" + html);
		myString jspath = myfs::path(project_path, "ui/" + js);
		myString csspath = myfs::path(project_path, "ui/ui.css");
		myString libpath = myfs::path("script/web/lib/");

		myParser::parse(inpath, outpath, {
			{"LIB_PATH",libpath},
			{"JS_PATH",jspath},
			{"CSS_PATH",csspath}
			});
		Navigate(view, "file://" + outpath);
	}

	void OnReady(myHandle id) override
	{
		navigate(menu, "menu.html", "menu.js");
		navigate(messageBox, "msg.html", "msg.js");


	}

	void OnMessageReceived(myHandle viewId, myString msg) override
	{
		json jmsg = json::parse(msg);

		myString id = jmsg["id"];
		myString value = jmsg["message"];
		if (id == "reload")
		{
			reloadModule();
		}

		if (id == "python")
		{
			debug << "Message from python: " << value << "\n";
			myWebView* messageBoxView = (myWebView*)getObject(messageBox);
			/*myString script = "const mynode = document.createTextNode('" + value + "');\n";
			script += "document.getElementById('message').appendChild(mynode)";

			messageBoxView->SetScript(script);*/
			messageBoxView->PostWebMessage(value);
		}

	}

	void OnKey(uint8_t key, bool pressed) override
	{
		if (pressed)
		{
			if (key == myKey::S) status = myPy::call("save_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
			if (key == myKey::R) reloadModule();
			if (key == myKey::L) status = myPy::call("load_", { myfs::path("user/DeepLearning/AutoEncoder/models/data/") });
		}


	}

	void run()
	{
		//status = myPy::call("runBatch", {}) >= 0;
	}

	void OnIdle() override
	{
		static int oldx = 0, oldy = 0;
		if (mousePressed) {
			if (mouseX != oldx || mouseY != oldy) {
				oldx = (int)mouseX;
				oldy = (int)mouseY;
				myPy::call("onMouseMove", { mouseX,mouseY });
			}
		}
	}
	int clamp(float val, float max = 2000) {
		if (val < 0) return 0;
		if (val > max) return (int)max;
		return (int)val;
	}


	void OnDraw() override
	{
		//myEngine::OnDraw();
		//Py::call("Forward2", {(int)mouseX,(int)mouseY});	
		static int oldx, oldy;
		int x = (int)mouseX;
		int y = (int)mouseY;
		
		oldx = x;
		oldy = y;
	}

	void OnUpdate() override
	{

	}

};
