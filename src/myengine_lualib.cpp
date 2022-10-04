#include "my.h"
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
static int engine_getversion(lua_State* L) {
	lua_pushstring(L, myEngine::pEngine->version);
	return 1;
}

static int engine_addWindow(lua_State* L) {

	int width = (int) luaL_checkinteger(L, 1);
	int height = (int) luaL_checkinteger(L, 2);
	int pixelWidth = (int) luaL_checkinteger(L, 3);
	int pixelHeight = (int) luaL_checkinteger(L, 4);
	int fullscreen = lua_toboolean(L, 5);
	int result = myEngine::pEngine->AddMainWindow(width, height, pixelWidth, pixelHeight, fullscreen);
	lua_pushnumber(L, result);
	return 1;
}

static int engine_clear(lua_State* L) {
	int color = (int) luaL_checkinteger(L, 1);
	myEngine::pEngine->Clear(myColor(color));
	return 0;
}

static int engine_setpixel(lua_State* L) {
	int x = (int) luaL_checkinteger(L, 1);
	int y = (int) luaL_checkinteger(L, 2);
	int color = (int) luaL_checkinteger(L, 3);
	if ((x >= 0) && (x < myEngine::pEngine->clientWidth) && (y >= 0) && (y < myEngine::pEngine->clientHeight))
		myEngine::pEngine->GetLinePointer(y)[x] = color;
	return 0;
}

static int engine_getpixel(lua_State* L) {
	int x = (int) luaL_checkinteger(L, 1);
	int y = (int) luaL_checkinteger(L, 2);
	int color = (int) luaL_checkinteger(L, 3);
	if ((x >= 0) && (x < myEngine::pEngine->clientWidth) && (y >= 0) && (y < myEngine::pEngine->clientHeight))
		lua_pushnumber(L,myEngine::pEngine->GetLinePointer(y)[x].n);
	return 1;
}

static int engine_drawhline(lua_State* L) {
	int x1 = (int)luaL_checkinteger(L, 1);
	int x2 = (int)luaL_checkinteger(L, 2);
	int y = (int)luaL_checkinteger(L, 3);
	int color = (int)luaL_checkinteger(L, 4);
	myEngine::pEngine->DrawHLine(x1, x2, y, color);
	return 0;
}

static int engine_drawvline(lua_State* L) {
	int y1 = (int)luaL_checkinteger(L, 1);
	int y2 = (int)luaL_checkinteger(L, 2);
	int x = (int)luaL_checkinteger(L, 3);
	int color = (int)luaL_checkinteger(L, 4);
	myEngine::pEngine->DrawHLine(y1, y2, x, color);
	return 0;
}

static int engine_fillrect(lua_State* L) {
	int x = (int)luaL_checkinteger(L, 1);
	int y = (int)luaL_checkinteger(L, 2);
	int w = (int)luaL_checkinteger(L, 3);
	int h = (int)luaL_checkinteger(L, 4);	
	int color = (int)luaL_checkinteger(L, 5);
	myEngine::pEngine->FillRect(ivec2(x, y), ivec2(w, h), color);
	return 0;
}

static int engine_drawrect(lua_State* L) {
	int x = (int)luaL_checkinteger(L, 1);
	int y = (int)luaL_checkinteger(L, 2);
	int w = (int)luaL_checkinteger(L, 3);
	int h = (int)luaL_checkinteger(L, 4);
	int color = (int)luaL_checkinteger(L, 5);
	myEngine::pEngine->DrawRect(ivec2(x, y), ivec2(w, h), color);
	return 0;
}

static int engine_drawline(lua_State* L) {
	int x1 = (int)luaL_checkinteger(L, 1);
	int y1 = (int)luaL_checkinteger(L, 2);
	int x2 = (int)luaL_checkinteger(L, 3);
	int y2 = (int)luaL_checkinteger(L, 4);	
	int color = (int)luaL_checkinteger(L, 5);
	myEngine::pEngine->DrawLine(ivec2(x1, y1), ivec2(x2, y2), color);
	return 0;
}


static int engine_fillcircle(lua_State* L) {
	int x = (int) luaL_checkinteger(L, 1);
	int y = (int) luaL_checkinteger(L, 2);
	int r = (int) luaL_checkinteger(L, 3);
	int color = (int) luaL_checkinteger(L, 4);
	myEngine::pEngine->FillCircle(ivec2(x, y), r, color);
	return 0;
}

static int engine_getmouse(lua_State* L) {
	lua_pushnumber(L, myEngine::pEngine->mouseX);
	lua_pushnumber(L, myEngine::pEngine->mouseY);	
	return 2;
}


static const struct luaL_Reg enginelib[] = {
{"getversion",	engine_getversion},
{"addwindow",	engine_addWindow},
{"clear",		engine_clear},
{"setpixel",	engine_setpixel},
{"getpixel",	engine_getpixel},
{"drawhline",	engine_drawhline},
{"drawvline",	engine_drawvline},
{"fillrect",	engine_fillrect},
{"drawrect",	engine_drawrect},
{"fillcircle",	engine_fillcircle},
{"getmouse",	engine_getmouse},
{NULL, NULL} /* sentinel */
};

int luaopen_mylib(void* L) {
	luaL_newlib((lua_State*)L, enginelib);
	return 1;
}

myLuaCFunction myEngine::getLibFunction()
{
	return luaopen_mylib;
}