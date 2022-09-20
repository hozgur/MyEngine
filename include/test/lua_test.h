
#include "my.h"

using namespace My;
class MyEngine : public My::myEngine
{
public:
	
	MyEngine(const char* path) :My::myEngine(path)
	{
		SetScript(myfs::path("user/lua_test.lua"));
	}

	bool OnStart() override
	{
		return true;
	}
};