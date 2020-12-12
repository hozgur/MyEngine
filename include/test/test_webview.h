#pragma once
#include "mypy.h"
using namespace My;

class MyEngine : public My::Engine
{
public:
    handle hWeb = invalidHandle;
        
    MyEngine(const char* path) :My::Engine(path)
    {        
        SetScript(myfs::path("user/lua_test.lua"));
        py.init();
        SetScript(myfs::path("user/webviewpy_test.py"));
    }

    bool OnStart() override
    {
        //AddWindow(1200, 800);
        hWeb = AddWebView(0, 0, 400, 300);
        return true;
    }

    void OnReady(handle id) override
    {
        //Navigate(id, "google.com.tr");
        const char* content =   "<!DOCTYPE html>"
                                "<html lang = 'en'>" 
                                "<head>"
                                "<style type = 'text/css' >"
                                "html{"
                                 "   margin : 0;"
                                 "   padding : 0;"
                                 "   background: rgba(0, 0, 0, .1);"
                                 "   color: blue;"
                                    "}"
                             "   </style>"
                                    "<meta charset = 'UTF-8'>"
                                    "<meta name = 'viewport' content = 'width = device - width, initial - scale = 1.0'>"
                                    "<title>Document</title>"
                                "</head>"
                                "<body>"
                                    "<h1>Hello from Content</h1>"
                                "</body>"
                                "</html>";



        NavigateContent(id, { content });
    }

    void OnMessageReceived(std::string message)
    {
        debug << message << "\n";
    }

    void OnDraw() override
    {
        Engine::OnDraw();
        /*if (view->IsReady() && KeyState[Key::A])
        {
            navigated = true;
            view->Navigate("http://www.google.com");
        }*/
    }

};