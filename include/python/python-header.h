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
	uint64_t initpyLastWriteTime;
	uint64_t runpyLastWriteTime;

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
		
		initpyLastWriteTime = loadModule("init");
		if (initpyLastWriteTime < 0) {
			exit(1);
		}		
		runpyLastWriteTime = loadModule("run");
		return true;
	}

	uint64_t loadModule(myString moduleName) {
		myString filePath = myfs::path(project_path, moduleName + ".py");
		if (!myPy::dofile(filePath)) {
			debug << "Error on running <" + moduleName+".py>\n";
			return -1;
		}
		return std::filesystem::last_write_time(filePath).time_since_epoch().count();		
	}

	bool loadModuleIfChanged(myString moduleName, uint64_t& lastWriteTime) {
		myString filePath = myfs::path(project_path, moduleName + ".py");
		uint64_t newWriteTime = std::filesystem::last_write_time(filePath).time_since_epoch().count();
		if (newWriteTime > lastWriteTime) {
			lastWriteTime = newWriteTime;
			if (!myPy::dofile(filePath)) {
				debug << "Error on running <" + moduleName + ".py>\n";
				return false;
			}
			return true;
		}
		return false;
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
			if (key == myKey::R) loadModule("run");
	}

	void run()
	{
		
	}

	void OnIdle() override
	{
		// check files for changes and reload if needed on time interval (100 msec)
		static uint64_t lastTime = 0;
		uint64_t now = myTime::now();
		if (now - lastTime > 100000) {
			lastTime = now;
			if (loadModuleIfChanged("init", initpyLastWriteTime)) loadModule("run");
			loadModuleIfChanged("run", runpyLastWriteTime);
		}
		
	}
	
	void OnDraw() override
	{
		
	}

	void OnUpdate() override
	{

	}

};
