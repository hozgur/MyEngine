#pragma once
namespace My
{
	typedef int (*mylua_CFunction) (void* L);
	class lualib
	{
	public:
		virtual mylua_CFunction getLibFunction() = 0;
	};
}
