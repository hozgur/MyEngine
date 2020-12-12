#pragma once
#pragma once
#include "mypy.h"
using namespace My;

class MyEngine : public My::Engine
{
public:
    handle hWeb = invalidHandle;
    std::thread bokehThread;
    MyEngine(const char* path) :My::Engine(path)
    {        
    }

    void bokehThreadFunc()
    {
        std::string bokehPath = myfs::getEnv("PyTHONHOME") + std::string("\\Scripts\\bokeh.exe serve ") + myfs::path("user/webview/bokeh_test.py");
        system(bokehPath.c_str());
    }
    bool OnStart() override
    {
        AddWindow(1200, 800);
        hWeb = AddWebView(0, 0, 800, 400);
        py.init();
        //SetScript(myfs::path("user/webview/bokeh_test.py"));
        bokehThread = std::thread(&MyEngine::bokehThreadFunc, this);
        return true;
    }
    
    void OnMessageReceived(std::string message)
    {
        debug << message << "\n";
    }

    void OnKey(uint8_t key, bool pressed) override
    {
        Navigate(hWeb, "http://localhost:5006/bokeh_test");
    }
    void OnReady()
    {
        Navigate(hWeb, "http://localhost:5006/bokeh_test");
    }

    void OnExit()
    {
        //bokehThread.join();
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