// engine.cpp
#include "my.h"
#include "mypy.h"
#include "windows\windowsplatform.h"
namespace fs = std::filesystem;
std::atomic<bool> myEngine::baThreadActive = { false };
myEngine* myEngine::pEngine = nullptr;
myEngine::myEngine(const char* path)
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
myObject* myEngine::getObject(myHandle id)
{
    if (id <= 0) return nullptr;
    std::map<myHandle, myObject*>::iterator it;
    it = objects.find(id);
    if (it == objects.end())
        return nullptr;
    else
        return it->second;
}
myHandle myEngine::setObject(myObject* obj)
{
    if (obj == nullptr) return -1;
    myHandle key = getHashCode();
    obj->SetID(key);
    objects[key] = obj;
    return key;
}
myEngine::~myEngine()
{
    for(std::map<myHandle, myObject*>::iterator it = objects.begin(); it != objects.end(); ++it)
        delete it->second;
    myPy::exit();
    delete background;
    delete pPlatform;
}

bool myEngine::AddWindow(int width,int height, int pixelWidth, int pixelHeight, bool fullScreen)
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

bool myEngine::Start()
{   
    if(myPy::isInitialized())
        if (!myPy::dofile(myfs::path("script\\init.py")))
            debug << "Python init.py Error\n";
    if (OnStart() == false) return false;
    pPlatform->StartUp();
    pPlatform->StartSystemEventLoop();                
    OnExit();//TODO: Lua implimentation
    pPlatform->CleanUp();
    return true;
}

bool myEngine::SetScript(std::string scriptPath)
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
        if (!myPy::dofile(scriptPath))
        {
            debug << "Error on Script " << scriptPath << "\n";
            pyEnable = false;
            return false;
        }
        pyEnable = true;
    }
    return true;
}
    

void myEngine::OnIdle()
{
    myCommand q;
    if (commandQueue.pop(q))
    {
        switch (q.commandID)
        {
            case myCommands::Navigate: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->Navigate(std::get<std::string>(q.params[0])); break;
            case myCommands::NavigateContent: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->NavigateContent(std::get<std::string>(q.params[0])); break;
            case myCommands::SetScript: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->SetScript(std::get<std::string>(q.params[0])); break;
            case myCommands::PostWebMessage: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->PostWebMessage(std::get<std::string>(q.params[0])); break;
        }
    }
}

void myEngine::Navigate(myHandle id, std::string uri)
{
    commandQueue.push({ id, myCommands::Navigate, {uri} });
}
void myEngine::NavigateContent(myHandle id, std::string content)
{
    commandQueue.push({ id, myCommands::NavigateContent, {content} });
}
void myEngine::SetScript(myHandle id, std::string scriptContent)
{
    commandQueue.push({ id, myCommands::SetScript, {scriptContent} });
}
void myEngine::PostWebMessage(myHandle id, std::string message)
{
    commandQueue.push({ id, myCommands::PostWebMessage, {message} });
}
    
myHandle myEngine::getHashCode()
{
    static myHandle key = 0;
    static bool lookup = false;

    if (lookup)
    {        
        while (objects.count(key) > 0)
        {
            key++;
            if (key == INT32_MAX) key = 0;
        }
    }

    if (key == INT32_MAX)
    {
        key = 0;
        lookup = true;
    }
    return key++;
}

myHandle myEngine::SetObject(myObject* obj)
{
    if (obj == nullptr) return -1;
    myHandle key = getHashCode();
    obj->SetID(key);
    objects[key] = obj;
    return key;
}

void myEngine::removeObject(myHandle id)
{
    delete objects[id];
    objects.erase(id);
}

void myEngine::EngineThread()
{
        
}
    
void myEngine::SetFPS(int fps)
{
    pPlatform->SetFPS(fps);
}

    
void myEngine::OnDraw()
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
        
    if (pyEnable && myPy::checkfunction("OnDraw"))
    {
        if (myPy::dofunction("OnDraw", {}) == false)
            pyEnable = false;
    }

}
void myEngine::OnUpdate()
{
    if (luaEnable && lua.checkfunction("OnUpdate"))
        if (lua.dofunction("OnUpdate") == false)
        {
            luaEnable = false;
            debug << lua.error();
        }        
}

void myEngine::OnExit()
{

}

void myEngine::OnKey(uint8_t key, bool pressed)
{
    KeyState[key] = pressed;
    if(pressed)
        pressedKey = key;
    else
        releasedKey = key;

}
void myEngine::OnMouse(myMouseEvent event, float x, float y)
{
}
bool myEngine::OnStart() { return true; }
        
void myEngine::SetWindowTitle(std::string title)
{
    pPlatform->SetWindowTitle(title);
}

void myEngine::UpdateKeyState(uint8_t key, bool state)
{
    KeyState[key] = state;
    OnKey(key, state);
}

myHandle myEngine::AddWebView(int x, int y, int width, int height)
{
    return SetObject(pPlatform->AddWebView(x, y, width, height));
}

myHandle myEngine::loadImage(std::string path)
{
    return SetObject(pPlatform->loadImage(path));
}
