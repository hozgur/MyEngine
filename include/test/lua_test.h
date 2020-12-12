
#include "my.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	
	MyEngine(const char* path) :My::Engine(path)
	{
		SetScript(myfs::path("user/lua_test.lua"));
	}

	bool OnStart() override
	{
		return true;
	}
};