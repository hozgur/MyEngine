#pragma once

struct myLua
{
	void* _L;
	myLua();
	~myLua();

	bool dofile(std::string file);		
	bool dostring(std::string content);
	bool loadstring(std::string file);		
	bool dofunction(std::string funcname, std::vector<variant> parameters = {});
	bool checkfunction(std::string funcname);
	bool loadlibrary(std::string libname, myLualib* library);
	template<typename T>
	T getglobal(const char* name){T result;return result;}
		
	template<>int getglobal<int>(const char* name){ return getglobalint(name);}

	std::string error();
	void stackdump();

	int getglobalint(const char* name);
	void setglobal(const char* name, int value);
	void setglobal(const char* name, float value);
};
	
extern myLua lua;