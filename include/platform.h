#pragma once

namespace My
{
    class Platform
    {
    public:

        virtual bool Init() = 0;
        virtual bool AddWindow(int width, int height, int pixelWidth = 1,int pixelHeight = 1, bool fullScreen = false) = 0;
        virtual webview* AddWebView(int x, int y, int width, int height) = 0;
        virtual void StartSystemEventLoop() = 0;
        virtual void SetFPS(int fps) = 0;
        virtual Color* GetLinePointer(int nLine) = 0;
        virtual void SetWindowTitle(std::string title) = 0;
        virtual int GetScanLine() = 0;
        virtual void Clear(Color c) = 0;
        virtual void StartUp() = 0;
        virtual void CleanUp() = 0;
    };
}
