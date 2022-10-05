// my.cpp
#include "python/python-header.h"

int main(int argc, char *argv[])
{   
	if (argc > 1) {
		myString mode = argv[1];
		if (mode == "python") {
			pyEngine engine((const char*)argv[0], argc, (const char**)argv);
			engine.Start();
			return 0;			
		}
		else {
			debug << "Unknown mode: " << mode << std::endl;
			debug << "Usage:myengine python project-path <path-to-python-project>\n";
			return 1;
		}
	}
	else
	{
		debug << "Please specify a mode. \n";
		debug << "Usage:myengine python project-path <path-to-python-project>\n";
		return 1;
	}	
}
