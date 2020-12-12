#pragma once
#include <ddraw.h>
#include <gdiplus.h>
#include "Windows\surface.h"
namespace My
{
	class WindowsPlatform : public Platform
	{
		HWND                hWnd = nullptr;
		UINT_PTR            hTimer = NULL;
		Surface* backSurface = nullptr;
		int                 nFPS = 60;
		ULONG_PTR           g_pDD = NULL;
		ULONG_PTR           m_gdiplusToken = NULL;
		StopWatch			sFPS;
		std::thread paintThread;
		static std::atomic<bool> ThreadActive;
	public:
		WindowsPlatform();
		~WindowsPlatform();

		virtual bool Init() override;
		virtual bool AddWindow(int width, int height, int pixelWidth = 1, int pixelHeight = 1, bool fullScreen = false) override;
		virtual webview* AddWebView(int x, int y, int width, int height) override;
		virtual void StartSystemEventLoop() override;
		virtual void SetFPS(int fps);
		inline virtual Color* GetLinePointer(int nLine);
		virtual int GetScanLine() override;
		virtual void SetWindowTitle(std::string title);
		void MapKeys();
		virtual void StartUp() override;
		virtual void CleanUp() override;
		void CreateWebViews();
		void CloseWebViews();
		void mainThread();

		virtual void OnPaint();
		virtual void Clear(Color c);
		//Static Functions		
		static LRESULT CALLBACK WindowEvent(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
	};
}

