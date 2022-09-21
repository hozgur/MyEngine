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
	int dofunction(std::string funcname, paramlist parameters);
	bool checkfunction(std::string funcname);		
	template<typename T>
	T getglobal(const char* name) { return {}; }
	template<typename T> void setglobal(const char* name, const T& val);
};	