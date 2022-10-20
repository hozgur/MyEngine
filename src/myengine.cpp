// engine.cpp
#include "my.h"
#include "mypy.h"
#include "windows\windowsplatform.h"

namespace fs = std::filesystem;
std::atomic<bool> myEngine::baThreadActive = { false };
myEngine* myEngine::pEngine = nullptr;
const char* myEngine::version = "3.0.0";
myEngine::myEngine(const char* path)
{
    appPath = path;
    luaEnable = false;
    pyEnable = false;
    pEngine = this;
    pPlatform = new WindowsPlatform();
    keepAliveonDestroyWindow = false;
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
void myEngine::removeObject(myHandle id)
{
    if (id > 0) {
        delete objects[id];
        objects.erase(id);
    }
}
myHandle myEngine::getHashCode()
{
    static std::mutex mtx;
    static const int keyStart = 1; // Eskiden 0 idi ama Python i�in 0 kullan�lmamas� gerekti�i i�in ba�lang�� 1 yap�ld�.
    static myHandle key = keyStart;
    static bool lookup = false;
    std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
    lck.lock();
    if (lookup)
    {
        while (objects.count(key) > 0)
        {
            key++;
            if (key == INT32_MAX) key = keyStart;
        }
    }

    if (key == INT32_MAX)
    {
        key = keyStart;
        lookup = true;
    }
    lck.unlock();
    return key++;
}



myEngine::~myEngine()
{
    DestroyMainWindow();
    myPy::exit();
    
    delete pPlatform;
}

bool myEngine::AddMainWindow(int width,int height, int pixelWidth, int pixelHeight, bool fullScreen)
{
    if ((width <= 0) || (height <= 0) || (pixelWidth <= 0) || (pixelHeight <= 0))
    {
        debug << "AddMainWindow - invalid parameters\n";
        return false;
    }
	if(clientWidth > 0) {
		keepAliveonDestroyWindow = true;
        DestroyMainWindow();
        keepAliveonDestroyWindow = false;
    }
    clientWidth = width;
    clientHeight = height;
    this->pixelWidth = pixelWidth;
    this->pixelHeight = pixelHeight;
    return pPlatform->AddMainWindow(width*pixelWidth,height*pixelHeight,pixelWidth,pixelHeight,fullScreen);
}

bool myEngine::DestroyMainWindow() {
	// Delete all objects
    for (std::map<myHandle, myObject*>::iterator it = objects.begin(); it != objects.end(); ++it)        
        delete it->second;
    objects.clear();
    childViews.clear();
    delete background;
	background = nullptr;	
	return pPlatform->DestroyMainWindow();
}

bool myEngine::Start()
{   
    if(myPy::isInitialized())
        if (!myPy::dofile(myfs::path("script\\init.py")))
            debug << "Python init.py Error\n";
	
    bool stat = OnStart();
    onIdle();   // for debug messages;
    if (stat == false) return false;
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

void myEngine::OnIdle() {
	
}

void myEngine::onIdle()
{
    myCommand q;
    if (commandQueue.pop(q))
    {
        switch (q.commandID)
        {
            case myCommands::Navigate: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->Navigate(std::get<std::string>(q.params[0])); break;
            case myCommands::NavigateContent: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->NavigateContent(std::get<std::string>(q.params[0])); break;
            case myCommands::SetScript: if (q.params.size() == 1) ((myWebView*)getObject(q.id))->SetScript(std::get<std::string>(q.params[0])); break;
            case myCommands::PostWebMessage: if (q.params.size() == 1) {
				myWebView* wv = (myWebView*)getObject(q.id);
				if (wv) wv->PostWebMessage(std::get<std::string>(q.params[0]));                
            }                
        }
    }
	
    if (debugStringStream.str().length() > 0) {
        if (debugHandlers.size() > 0)
        {
            std::string s = debugStringStream.str();
            for (auto& h : debugHandlers)
            {
                h(s);
            }            
        }
		std::cout << debugStringStream.str();
        debugStringStream.str("");
    }
	
    OnIdle();
}
void myEngine::onSize(int cx,int cy) {
	OnSize(cx, cy); // send before clientWidth and Height changes.
	int dx = cx - clientWidth;
	int dy = cy - clientHeight;
    clientWidth = cx;
    clientHeight = cy;
    for (myHandle viewHandle : childViews) {		
        myView* view = (myView*)getObject(viewHandle);
        if (view) {
            // left - right
            if (view->anchor & myAnchorLeft) {
                if (view->anchor & myAnchorRight) {
                    int width = 0, height = 0;
                    view->GetSize(width, height);
                    view->SetSize(clientWidth, height);
                }
            }
            else {
                if (view->anchor & myAnchorRight) {
                    int x = 0, y = 0;
                    view->GetPosition(x, y);
                    x += dx;
                    view->SetPosition(x, y);
                }
            }
            // top - bottom
            if (view->anchor & myAnchorTop) {
                if (view->anchor & myAnchorBottom) {
                    int width = 0, height = 0;
                    view->GetSize(width, height);
                    view->SetSize(width, clientHeight);
                }
            }
            else {
                if (view->anchor & myAnchorBottom) {
                    int x = 0, y = 0;
                    view->GetPosition(x, y);
                    y += dy;
                    view->SetPosition(x, y);
                }
            }
        }
        else {
            debug << "There is no invalid view! ViewHandle = " << viewHandle << "\n";
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
        if (myPy::call("OnDraw", {}) == false)
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
void myEngine::OnMouse(myMouseEvent event, float x, float y) {
}
void myEngine::OnSize(int cx, int cy) {
}
bool myEngine::OnStart() { return true; }
        
void myEngine::SetWindowTitle(std::string title) {
    pPlatform->SetWindowTitle(title);
}

void myEngine::UpdateKeyState(uint8_t key, bool state) {
    KeyState[key] = state;
    OnKey(key, state);
}

myHandle myEngine::AddWebView(int x, int y, int width, int height, int anchor) {
    myView* view = pPlatform->AddWebView(x, y, width, height,(myAnchor)anchor);
    if (view) {
        myHandle id = setObject(view);
        childViews.push_back(id);
        return id;
    }
    return invalidHandle;
}

myHandle myEngine::loadImage(std::string path)
{
    return setObject(pPlatform->loadImage(path));
}
