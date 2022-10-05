#pragma once
#include "my.h"
#include "mypy.h"
#include <format>
void usage() {
	debug << "Usage:myengine python project-path <path-to-project>\n" << "Example:myengine project-path projects/python/graphics\n";
}

class pyEngine : public myEngine
{
public:
	
	myString project_path;
	myString initFilePath;
	myString runFilePath;
	myTime initLastTime;
	myTime runLastTime;
	
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
			debug << "Unknown argument: " << arg << std::endl;
			usage();
			exit(1);
		}
		initFilePath = myfs::path(project_path, "init.py");
		runFilePath = myfs::path(project_path, "run.py");
		initLastTime = myfs::lastWriteTime(initFilePath);;
		runLastTime = myfs::lastWriteTime(runFilePath);;
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
		
		init();
		run();
		return true;
	}
	
	void navigate(myHandle view, myString html, myString js) {
		
	}

	void init() {
		if (myfs::exists(initFilePath)) {
			myPy::dofile(initFilePath);
			initLastTime = myfs::lastWriteTime(initFilePath);
		}
	}

	void run() {
		if (myfs::exists(runFilePath)) {
			myPy::dofile(runFilePath);
			runLastTime = myfs::lastWriteTime(runFilePath);
		}
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
	
	}

	void OnEveryMiliseconds(int mils,std::function<void(void)> function) {
		static myTime lastTime = 0;
		myTime now = myos::now()/1000;
		if (now - lastTime > mils) {
			lastTime = now;
			function();
		}
	}

	void checkFileChanges() {
		if (myfs::lastWriteTime(initFilePath) > initLastTime) {
			init(); run();
		}
		if (myfs::lastWriteTime(runFilePath) > runLastTime) {
			run();
		}
	}
	
	void OnIdle() override
	{		
		OnEveryMiliseconds(100, [&]() {
			checkFileChanges();
			});		
	}
	
	void OnDraw() override
	{
		
	}

	void OnUpdate() override
	{

	}

};
