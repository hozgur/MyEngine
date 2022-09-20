#pragma once
#include "my.h"
#include "python/pymodule.h"

namespace myPy
{	
	typedef std::variant<long, double, std::string> pyvariant;
	typedef std::pair<std::string, pyvariant> dictItem;
	typedef std::map<std::string, pyvariant> dict;
	typedef std::vector<pyvariant> paramlist;
	extern wchar_t* program;
		
	bool init();
	void exit();
	bool isInitialized();
	bool dofile(std::string file);
	bool addModule(myPyModule* module);		
	bool dostring(std::string content);
	bool dostring(std::string content, dict locals);
	bool dostring(std::string content, dict locals, dict &result);		
	bool dofunction(std::string funcname, paramlist parameters);
	bool checkfunction(std::string funcname);		
	template<typename T>
	T getglobal(const char* name) { T result; return result; }

	//template<>int getglobal<int>(const char* name) { return getglobalint(name); }

	//std::string error();
		

	//int getglobalint(const char* name);
};	