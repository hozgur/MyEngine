#pragma once
#include "mypy.h"
#include "myparser.h"

class MyEngine : public myEngine
{
public:
    myHandle hWeb = invalidHandle;
    const int width = 800;
    const int height = 600;
    const int posX = 400;
    const int posY = 0;
    int size = 7;
	float speed = 1;
    MyEngine(const char* path) :myEngine(path)
    {
        
        SetScript(myfs::path("user/lua_test.lua"));
        lua.setglobal("clientWidth", width);
        lua.setglobal("clientHeight", height);
				
        myPy::init();
        SetScript(myfs::path("user/webviewpy_test.py"));
    }

    bool OnStart() override
    {
		
        AddWindow(width, height);
        hWeb = AddWebView(posX, posY, 400, 300);
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

    
    virtual void OnMessageReceived(myHandle id, std::string message) override
    {
        //debug << message;
        json msg = json::parse(message);
		
        if (msg["event"] == "mousemove") {
            myEngine::pEngine->mouseX = (float) (msg["x"] + posX);
            myEngine::pEngine->mouseY = (float) (msg["y"] + posY);
        }
        if (msg["event"] == "sizechange") {            
			size = msg["value"];
            debug << "size = " << size <<"\n";
        }
        if (msg["event"] == "speedchange") {            
            speed = msg["value"];
            debug << "speed = " << speed << "\n";
        }
    }
    void OnDraw() override
    {
		lua.setglobal("circleSize", size);
        lua.setglobal("speed", speed);
        myEngine::OnDraw();
        /*if (view->IsReady() && KeyState[myKey::A])
        {
            navigated = true;
            view->Navigate("http://www.google.com");
        }*/
    }

};