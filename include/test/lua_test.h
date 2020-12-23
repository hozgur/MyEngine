
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
		AddWindow(1200, 800);
		SetScript(myfs::path("user/lua_test.lua"));
		return true;
	}
};