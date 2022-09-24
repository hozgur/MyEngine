#pragma once
// engine.h
#include "mylualib.h"
#include "Eigen/Dense"

using Eigen::Matrix;
typedef Matrix<float, 2, 1> vec2;
typedef Matrix<int, 2, 1> ivec2;
typedef void (*drawLineFunc)(int,int,myColor);
enum class myMouseEvent {Mouse_LBPressed, Mouse_RBPressed, Mouse_MBPressed, Mouse_LBReleased, MouseRBReleased, Mouse_MBReleased,
                        Mouse_Move};
enum myKey
{
    NONE,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    K0, K1, K2, K3, K4, K5, K6, K7, K8, K9,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    UP, DOWN, LEFT, RIGHT,
    SPACE, TAB, SHIFT, CTRL, INS, DEL, HOME, END, PGUP, PGDN,
    BACK, ESCAPE, RETURN, ENTER, PAUSE, SCROLL,
    NP0, NP1, NP2, NP3, NP4, NP5, NP6, NP7, NP8, NP9,
    NP_MUL, NP_DIV, NP_ADD, NP_SUB, NP_DECIMAL, PERIOD
};
class myEngine : public myLualib, myView
{
public:
    const char* version = "1.0.0";
    myPlatform* pPlatform;
    int clientWidth;
    int clientHeight;
    int monitorWidth;
    int monitorHeight;
    int pixelWidth;
    int pixelHeight;
    std::atomic<float> mouseX;
    std::atomic<float> mouseY;    
    bool mousePressed;
    static std::atomic<bool> baThreadActive;
    static myEngine* pEngine;
    std::string appPath;
    bool luaEnable;
    bool pyEnable;
    bool KeyState[256] = { 0 };
    int pressedKey = -1;
    int releasedKey = -1;
    std::map<myHandle, myObject*> objects;
    std::vector<myHandle> childViews;
    myCommandQueue commandQueue;
    myImage<myColor>* background = nullptr;
    public:
    myEngine(const char* path);
    virtual ~myEngine();
    bool Start();
    bool SetScript(std::string scriptPath);
    bool AddWindow(int width,int height, int pixelWidth = 1, int pixelHeight = 1, bool fullScreen = false);
    myHandle AddWebView(int x,int y, int width, int height,int anchor = myAnchorNone);
    int GetScanLine() { return pPlatform->GetScanLine(); }
    myColor* GetLinePointer(int nLine) { return pPlatform->GetLinePointer(nLine); }
    myColor& Pixel(int x, int y) { static myColor color; if (isInScreen(x, y)) return GetLinePointer(y)[x]; else return color; }
    bool isInScreen(int x, int y) { return (x >= 0) && (x < clientWidth) && (y >= 0) && (y < clientHeight); }
    void SetWindowTitle(std::string title);
    void SetFPS(int fps);

    bool GetKeyState(uint8_t key) { return KeyState[key]; }


    // Drawing Functions
    void Clear(myColor c,myHandle h = -1);
    void DrawHLine(int x1, int x2, int y, const myColor& c);
    void DrawVLine(int y1, int y2, int x, const myColor& c);
    void FillRect(ivec2 pos, ivec2 size, const myColor& c);
    void DrawRect(ivec2 pos, ivec2 size, const myColor& c);
    void FillCircle(ivec2 pos, int radius, const myColor& c);
    void DrawLine(int32_t x1, int32_t y1, int32_t x2, int32_t y2, myColor p, drawLineFunc func);
    void DrawLine(int32_t x1, int32_t y1, int32_t x2, int32_t y2, myColor p, uint32_t pattern = 0xFFFFFFFF);
    void DrawLine(ivec2 p1, ivec2 p2, myColor p, uint32_t pattern = 0xFFFFFFFF);
    void DrawText(int x, int y, std::string text,int fontHeight);
    void DrawImage(myHandle sourceImage, int x, int y, int width, int height, int sx, int sy,myHandle destImage = -1);
    //Image Functions

    myHandle loadImage(std::string path);
    template<typename T>
    myHandle createImage(int width, int height, int channelCount, Interleave interleave = Interleave::interleave_interleaved);

    // Handlers
    virtual void OnDraw();
    virtual void OnUpdate();
    virtual bool OnStart();
    virtual void OnExit();
    virtual void OnIdle();
    virtual void OnKey(uint8_t key, bool pressed);
    virtual void OnMouse(myMouseEvent event, float x, float y);
    virtual void OnSize(int cx, int cy);

    //WebView Methods
    void Navigate(myHandle id, std::string uri);
    void NavigateContent(myHandle id, std::string content);
    void SetScript(myHandle id, std::string scriptContent);
    void PostWebMessage(myHandle id, std::string message);

    //WebView Handlers

    virtual void OnReady(myHandle id) {};
    virtual bool OnNavigate(myHandle id, std::string uri) { return true; }
    virtual void OnMessageReceived(myHandle id, std::string message) { debug << "WebView Received Message = " << message << "\n"; }
    virtual void OnError(myHandle id, std::string uri) {};


    void        removeObject(myHandle id);        
    myObject*   getObject(myHandle id);
    myHandle    setObject(myObject* obj);    
    void UpdateKeyState(uint8_t key, bool state);

protected:
    void EngineThread();
    myHandle getHashCode();
    myHandle SetObject(myObject* obj);
	
//private:
public: //TODO: find a way to make this function private	
    virtual void onIdle();
    virtual void onSize(int cx, int cy);
public:
    template<typename T> void ForEachPixel(T&& lambda)
    {
        myColor* pp = GetLinePointer(0);
        int stride = (int)(GetLinePointer(1) - pp);
#pragma omp parallel for
        for (int y = 0; y < clientHeight; y++)
        {
            myColor* p = pp + y * stride;
            for (int x = 0; x < clientWidth; x++)
                lambda(x, y, p[x]);
        }
    }
    myLuaCFunction getLibFunction();

    // Inherited via myView
    virtual bool SetSize(int width, int height) override;
    virtual bool SetPosition(int x, int y) override;
    virtual bool GetSize(int& width, int& height) override;
    virtual bool GetPosition(int& x, int& y) override;    
    virtual bool SetAnchors(myAnchor anchors) override;
};
