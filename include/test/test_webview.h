#pragma once
#include "mypy.h"

class MyEngine : public myEngine
{
public:
    myHandle hWeb = invalidHandle;
        
    MyEngine(const char* path) :myEngine(path)
    {        
        SetScript(myfs::path("user/lua_test.lua"));
        myPy::init();
        SetScript(myfs::path("user/webviewpy_test.py"));
    }

    bool OnStart() override
    {
        AddWindow(1920, 1080);
        hWeb = AddWebView(1920-400, 0, 400, 300);
        return true;
    }

    void OnReady(myHandle id) override
    {
        std::string url = myfs::path("user/webview/compiled/index.html");
        url = "file://" + url;
        Navigate(id, url);
        /*const char* content =   "<!DOCTYPE html>"
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



        NavigateContent(id, { content });*/
    }

    void OnMessageReceived(std::string message)
    {
        debug << message << "\n";
    }

    void OnDraw() override
    {
        myEngine::OnDraw();
        /*if (view->IsReady() && KeyState[myKey::A])
        {
            navigated = true;
            view->Navigate("http://www.google.com");
        }*/
    }

};