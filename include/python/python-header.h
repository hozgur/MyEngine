#pragma once
#include "my.h"
#include "mypy.h"

void usage() {
	debug << "Usage:myengine python project-path <path-to-project>\n" << "Example:myengine project-path projects/python/graphics\n";
}

class pyEngine : public myEngine
{
public:
	
	myString project_path;
	

	pyEngine(const char* path,const int argc, const char *argv[]) :myEngine(path)
	{
		if (argc < 4) {
			usage();
			exit(1);
		}
		myString arg = argv[2];
		if (arg == "project-path") {			
			project_path = argv[3];
		}
		else {
			usage();
			exit(1);
		}
	}
			
	bool OnStart() override
	{
		if (myPy::init()) {
			debug << "Python ready. :)\n";			
		}
		else {
			debug << "Py Init Error!\n";
			exit(1);
		}
			
		if (!myPy::dofile(myfs::path(project_path, "init.py"))) {
			debug << "Error on init.py\n";
			exit(1);
		}		
		
		reloadModule();	// first load
		return true;
	}

	void reloadModule() {
		myPy::dofile(myfs::path(project_path, "run.py"));
	}

	void navigate(myHandle view, myString html, myString js) {
		
	}

	void OnReady(myHandle id) override
	{
		if (myPy::checkfunction("OnReady")) {			
			myPy::call("OnReady", { (long)id });
		}
		
	}

	void OnMessageReceived(myHandle senderId, myString msg) override
	{
		

	}

	void OnKey(uint8_t key, bool pressed) override
	{
		if (pressed)
			if (key == myKey::R) reloadModule();
	}

	void run()
	{
		
	}

	void OnIdle() override
	{
		
	}
	
	void OnDraw() override
	{
		
	}

	void OnUpdate() override
	{

	}

};
