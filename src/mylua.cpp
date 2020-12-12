
#include "my.h"
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

My::LUA My::lua;

#define L ((lua_State*)_L)


My::LUA::LUA()
{
	_L = luaL_newstate();
	luaL_openlibs(L);
}

My::LUA::~LUA()
{
	lua_close(L);
}

bool My::LUA::dofile(std::string file)
{
	return luaL_dofile(L,file.c_str()) == 0;
}

bool My::LUA::dostring(std::string content)
{
	return luaL_dostring(L, content.c_str()) == 0;
}

bool My::LUA::loadstring(std::string str)
{
    return luaL_loadstring(L, str.c_str()) == 0;
}

bool My::LUA::dofunction(std::string funcname, std::vector<variant> parameters)
{
    lua_getglobal(L, funcname.c_str());
        
    int paramCount = 0;
    for (auto const& value : parameters) {
        if (std::get_if<float>(&value))
        {
            float fvalue = std::get<float>(value);
            lua_pushnumber(L, fvalue );
            paramCount++;
        }

        if (std::get_if<int>(&value))
        {
            int ival = std::get<int>(value);
            lua_pushinteger(L, ival );
            paramCount++;
        }

        if (std::get_if<const char*>(&value))
        {
            lua_pushstring(L, std::get<const char*>(value));
            paramCount++;
        }
    }

    if (lua_pcall(L, paramCount, 1, 0) != 0) {
        debug << lua.error();
        return false;
    }
    int isnum;
    int result = (int)lua_tointegerx(L, -1, &isnum);
    if (!isnum) {
        debug << "function " << funcname << " should return number\n";
    }
    
    lua_pop(L, 1);

    return true;
                
}

bool My::LUA::checkfunction(std::string funcname)
{
    lua_getglobal(L, funcname.c_str());
    if (!lua_isfunction(L, -1))
    {
        lua_pop(L, 1);
        return false;
    }
    lua_pop(L, 1);
    return true;
}

bool My::LUA::loadlibrary(std::string libname, lualib* library)
{
    luaL_requiref(L, libname.c_str(), (lua_CFunction)library->getLibFunction(), 1);
    lua_pop(L, 1);  /* remove lib */
    return false;
}

std::string My::LUA::error()
{
    std::string err;
    if(lua_type(L,-1) == LUA_TSTRING)
	    err = lua_tostring(L, -1);
	lua_pop(L, 1);
	return err;
}

void My::LUA::stackdump()
{
    int i;
    int top = lua_gettop(L);
    for (i = 1; i <= top; i++) {  /* repeat for each level */
        int t = lua_type(L, i);
        switch (t) {

        case LUA_TSTRING:  /* strings */
            printf("`%s'", lua_tostring(L, i));
            break;

        case LUA_TBOOLEAN:  /* booleans */
            printf(lua_toboolean(L, i) ? "true" : "false");
            break;

        case LUA_TNUMBER:  /* numbers */
            printf("%g", lua_tonumber(L, i));
            break;

        default:  /* other values */
            printf("%s", lua_typename(L, t));
            break;

        }
        printf("  ");  /* put a separator */
    }
    printf("\n");  /* end the listing */
}

int My::LUA::getglobalint(const char* name)
{
    int isnum;
    lua_getglobal(L, name);
    int result = (int)lua_tointegerx(L, -1, &isnum);
    assert(isnum);
    lua_pop(L, 1);
    return result;
}
