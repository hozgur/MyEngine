// engine.cpp
#include "my.h"
#include "mypy.h"
#include "windows\windowsplatform.h"
namespace fs = std::filesystem;
namespace My
{
    std::atomic<bool> Engine::baThreadActive = { false };
    Engine* Engine::pEngine = nullptr;
    Engine::Engine(const char* path)
    {
        appPath = path;
        luaEnable = false;
        pyEnable = false;
        pEngine = this;
        pPlatform = new WindowsPlatform();
        lua.loadlibrary("engine", this);
        
        if (!lua.dofile(myfs::path("script\\init.lua")))
            debug << lua.error();
        
        clientWidth = clientHeight = 0;
        mouseX = 0; mouseY = 0;
        mousePressed = false;
        
    }

    Engine::~Engine()
    {
        Py::exit();
        if (objects.size() > 0)
        {
            debug << "There are " << objects.size() << " objects are still on memory.\n";
            for (std::map<handle, object*>::iterator it = objects.begin(); it != objects.end(); ++it)
                delete it->second;
        }
        delete background;
        delete pPlatform;
    }

    bool Engine::AddWindow(int width,int height, int pixelWidth, int pixelHeight, bool fullScreen)
    {
        if ((width <= 0) || (height <= 0) || (pixelWidth <= 0) || (pixelHeight <= 0))
        {
            debug << "AddWindow - invalid parameters\n";
            return false;
        }
        clientWidth = width;
        clientHeight = height;
        this->pixelWidth = pixelWidth;
        this->pixelHeight = pixelHeight;
        return pPlatform->AddWindow(width*pixelWidth,height*pixelHeight,pixelWidth,pixelHeight,fullScreen);
    }

    bool Engine::Start()
    {   
        if(Py::isInitialized())
            if (!Py::dofile(myfs::path("script\\init.py")))
                debug << "Python init.py Error\n";
        if (OnStart() == false) return false;
        pPlatform->StartUp();
        pPlatform->StartSystemEventLoop();                
        OnExit();//TODO: Lua implimentation
        pPlatform->CleanUp();
        return true;
    }

    bool Engine::SetScript(std::string scriptPath)
    {
        std::string ext = fs::path(scriptPath).extension().string();
        if (ext == ".lua")
        {
            if (!lua.dofile(scriptPath))
            {
                debug << lua.error();
                luaEnable = false;
                return false;
            }
            luaEnable = true;
            return true;
        }
        if (ext == ".py")
        {
            if (!Py::dofile(scriptPath))
            {
                debug << "Error on Script " << scriptPath << "\n";
                pyEnable = false;
                return false;
            }
            pyEnable = true;
        }
        return true;
    }
    

    void Engine::OnIdle()
    {
        command q;
        if (commandQueue.pop(q))
        {
            switch (q.commandID)
            {
                case Commands::Navigate: if (q.params.size() == 1) ((webview*)GetMyObject(q.id))->Navigate(std::get<std::string>(q.params[0])); break;
                case Commands::NavigateContent: if (q.params.size() == 1) ((webview*)GetMyObject(q.id))->NavigateContent(std::get<std::string>(q.params[0])); break;
                case Commands::SetScript: if (q.params.size() == 1) ((webview*)GetMyObject(q.id))->SetScript(std::get<std::string>(q.params[0])); break;
                case Commands::PostWebMessage: if (q.params.size() == 1) ((webview*)GetMyObject(q.id))->PostWebMessage(std::get<std::string>(q.params[0])); break;
            }
        }
    }

    void Engine::Navigate(handle id, std::string uri)
    {
        commandQueue.push({ id, Commands::Navigate, {uri} });
    }
    void Engine::NavigateContent(handle id, std::string content)
    {
        commandQueue.push({ id, Commands::NavigateContent, {content} });
    }
    void Engine::SetScript(handle id, std::string scriptContent)
    {
        commandQueue.push({ id, Commands::SetScript, {scriptContent} });
    }
    void Engine::PostWebMessage(handle id, std::string message)
    {
        commandQueue.push({ id, Commands::PostWebMessage, {message} });
    }
    
    handle Engine::GetHashCode()
    {
        static handle key = 1;  // Eskiden 0 idi ama Python için 0 kullanýlmamasý gerektiði için baþlangýç 1 yapýldý.
        static bool lookup = false;

        if (lookup)
        {        
            while (objects.count(key) > 0)
            {
                key++;
                if (key == INT32_MAX) key = 1;
            }
        }

        if (key == INT32_MAX)
        {
            key = 1;
            lookup = true;
        }
        return key++;
    }

    object* Engine::GetMyObject(handle id)
    {
        if (id <= 0) return nullptr;
        std::map<handle, object*>::iterator it;
        it = objects.find(id);
        if (it == objects.end())
            return nullptr;
        else
            return it->second;
    }

    handle Engine::SetMyObject(object* obj)
    {
        if (obj == nullptr) return -1;
        handle key = GetHashCode();
        obj->SetID(key);
        objects[key] = obj;        
        return key;
    }

    void Engine::RemoveMyObject(handle id)
    {
        if (id <= 0) return;
        std::map<handle, object*>::iterator it;
        it = objects.find(id);
        if (it != objects.end())
        {
            delete objects[id];
            objects.erase(id);
        }
        else
        {
            debug << "Object not found on Remove Object\n";
        }        
    }

    void Engine::EngineThread()
    {
        
    }
    
    void Engine::SetFPS(int fps)
    {
        pPlatform->SetFPS(fps);
    }

    
    void Engine::OnDraw()
    {
        if (luaEnable && (pressedKey >= 0) && lua.checkfunction("OnKey"))
        {            
            if (lua.dofunction("OnKey", {pressedKey,1}) == false)
            {
                luaEnable = false;
                debug << lua.error();
            }
            pressedKey = -1;
        }

        if (luaEnable && lua.checkfunction("OnDraw"))
            if (lua.dofunction("OnDraw") == false)
            {
                luaEnable = false;
                debug << lua.error();
            }
        
        /*if (pyEnable && Py::checkfunction("OnDraw"))
        {
            if (Py::dofunction("OnDraw", {}) == false)
                pyEnable = false;
        }*/

    }
    void Engine::OnUpdate()
    {
        if (luaEnable && lua.checkfunction("OnUpdate"))
            if (lua.dofunction("OnUpdate") == false)
            {
                luaEnable = false;
                debug << lua.error();
            }        
    }

    void Engine::OnExit()
    {

    }

    void Engine::OnKey(uint8_t key, bool pressed)
    {
        KeyState[key] = pressed;
        if(pressed)
            pressedKey = key;
        else
            releasedKey = key;

    }
    void Engine::OnMouse(MouseEvent event, float x, float y)
    {
    }
    bool Engine::OnStart() { return true; }
        
    void Engine::SetWindowTitle(std::string title)
    {
        pPlatform->SetWindowTitle(title);
    }

    void Engine::UpdateKeyState(uint8_t key, bool state)
    {
        KeyState[key] = state;
        OnKey(key, state);
    }

    handle Engine::AddWebView(int x, int y, int width, int height)
    {
        return SetMyObject(pPlatform->AddWebView(x, y, width, height));
    }

    handle Engine::loadImage(std::string path)
    {
        return SetMyObject(pPlatform->loadImage(path));
    }
}
