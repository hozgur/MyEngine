#pragma once
#include "my.h"
using namespace My;
#include "myparser.h"
#include <3rdParty/json.hpp>
//note: check quasar 
using json = nlohmann::json;

class MyEngine : public My::Engine
{
public:
    handle hView = invalidHandle;

    MyEngine(const char* path) :My::Engine(path)
    {
    }


    bool OnStart() override
    {
        SetScript(myfs::path("user/lua_test.lua"));
        int w = lua.getglobal<int>("clientWidth");
        int h = lua.getglobal<int>("clientHeight");
        int pw = lua.getglobal<int>("pixelWidth");
        int ph = lua.getglobal<int>("pixelHeight");


        int dw = lua.getglobal<int>("datWidth");
        int dh = lua.getglobal<int>("datHeight");
        int dx = lua.getglobal<int>("datX");
        int dy = lua.getglobal<int>("datY");

        AddWindow(w, h, pw, ph);
        //hView = AddWebView(dx, dy, dw, dh);
        hView = AddWebView(0, 0, w, h);
        //compile();
        return true;
    }

    void compile()
    {
        std::string inpath = myfs::path("user/webview/precompiled/vue_index2.html");
        std::string outpath = myfs::path("user/webview/compiled/vue_index2.html");
        std::string libpath = myfs::path("script/web/lib/");

        Parser::parse(inpath, outpath, {
            {"LIB_PATH",libpath},
            {"DAT_PATH","dat.gui.min.js"},
            {"JQUERY_PATH","jquery-3.5.1.min.js"},
            {"SEMANTIC_PATH","Semantic-UI/semantic.min.js"},
            });
    }


    void OnKey(uint8_t key, bool pressed) override
    {
        //compile();
        std::string path = myfs::path("user/webview/compiled/vue_index2.html");
        Navigate(hView, "file://" + path);
    }
    void OnReady(handle id) override
    {
        std::string path = myfs::path("user/webview/precompiled/vue_index2.html");
        Navigate(hView, "file://" + path);
    }

    

};