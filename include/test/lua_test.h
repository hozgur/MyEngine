
#include "my.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	
	MyEngine(const char* path) :My::Engine(path)
	{
		
	}

	bool OnStart() override
	{
		AddWindow(800, 600);
		SetScript(myfs::path("user/lua_test.lua"));
		return true;
	}
};