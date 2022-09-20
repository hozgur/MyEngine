
#include "my.h"
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

myLua lua;

#define L ((lua_State*)_L)


myLua::myLua()
{
	_L = luaL_newstate();
	luaL_openlibs(L);
}

myLua::~myLua()
{
	lua_close(L);
}

bool myLua::dofile(std::string file)
{
	return luaL_dofile(L,file.c_str()) == 0;
}

bool myLua::dostring(std::string content)
{
	return luaL_dostring(L, content.c_str()) == 0;
}

bool myLua::loadstring(std::string str)
{
    return luaL_loadstring(L, str.c_str()) == 0;
}

bool myLua::dofunction(std::string funcname, std::vector<variant> parameters)
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

bool myLua::checkfunction(std::string funcname)
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

bool myLua::loadlibrary(std::string libname, myLualib* library)
{
    luaL_requiref(L, libname.c_str(), (lua_CFunction)library->getLibFunction(), 1);
    lua_pop(L, 1);  /* remove lib */
    return false;
}

std::string myLua::error()
{
    std::string err;
    if(lua_type(L,-1) == LUA_TSTRING)
	    err = lua_tostring(L, -1);
	lua_pop(L, 1);
	return err;
}

void myLua::stackdump()
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

int myLua::getglobalint(const char* name)
{
    int isnum;
    lua_getglobal(L, name);
    int result = (int)lua_tointegerx(L, -1, &isnum);
    assert(isnum);
    lua_pop(L, 1);
    return result;
}
