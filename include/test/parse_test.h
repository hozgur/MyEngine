#pragma once
#include "my.h"
using namespace My;
#include "myparser.h"
class MyEngine : public Engine
{
public:

    MyEngine(const char* path) :Engine(path)
    {
    }

    bool OnStart() override
    {
        std::string inpath = myfs::path("user/webview/precompiled/index.html");
        std::string outpath = myfs::path("user/webview/compiled/index.html");
        std::string libpath = myfs::path("script/web/lib/");

        Parser::parse(inpath, outpath, {
            {"LIB_PATH",libpath},
            {"DAT_PATH","dat.gui.min.js"},
            {"JQUERY_PATH","jquery-3.5.1.min.js"},
            {"SEMANTIC_PATH","Semantic-UI/semantic.min.js"},
            });
        return true;
    }
    

};
