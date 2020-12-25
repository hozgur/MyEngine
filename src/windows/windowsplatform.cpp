#pragma once
#include "my.h"
#include "windows\windowscore.h"
#include "Windows\windowsplatform.h"
#include <Shlwapi.h>
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "ddraw.lib")
#include "Windows\directdraw.h"
#include "Windows\windowswebview.h"
#include "Windows\windowsimage.h"
namespace My
{	
	std::atomic<bool> WindowsPlatform::ThreadActive = { false };
	static std::map<size_t, uint8_t> mapKeys;
	WindowsPlatform::WindowsPlatform() :Platform()
	{
		Gdiplus::GdiplusStartupInput gdiplusstartupinput;
		Gdiplus::GdiplusStartup(&m_gdiplusToken, &gdiplusstartupinput, NULL);
		backSurface = nullptr;
		SetConsoleOutputCP(65001);
		Init();
	}

	WindowsPlatform::~WindowsPlatform()
	{
		if (g_pDD != NULL)
		{
			((LPDIRECTDRAW)g_pDD)->Release();
			g_pDD = NULL;
		}
		if (m_gdiplusToken != NULL)
		{
			Gdiplus::GdiplusShutdown(m_gdiplusToken);
		}
	}
	bool WindowsPlatform::Init()
	{

		HRESULT hr = DirectDrawCreate(NULL,(LPDIRECTDRAW*)&g_pDD, NULL);
		if (DDFailedCheck(hr, "DirectDrawCreate failed"))
			return false;
		MapKeys();
		return true;
	}

	void WindowsPlatform::StartUp()
	{
		if (hWnd != nullptr)
		{
			ThreadActive = true;
			paintThread = std::thread(&WindowsPlatform::mainThread, this);
		}		
	}

	void WindowsPlatform::CleanUp()
	{
		ThreadActive = false;
		if (paintThread.joinable())
			paintThread.join();
	}
	
	void WindowsPlatform::mainThread()
	{
		static int lastline = 0;
		static bool drawed = false;
		static StopWatch s;
		static bool first = true;
		
		while (ThreadActive)
		{
			int scanline = GetScanLine();
			if (scanline < 0)
			{
				if (first)
				{
					first = false;
					s.Start();
				}
				else
				{
					s.Stop();
					double d = s.GetDurationMS();
					if (d > 5)
					{
						s.Start();
						Engine::pEngine->OnDraw();
						OnPaint();
					}
				}
			}
			else
			{
				if (!drawed && (scanline > 700))
				{
					Engine::pEngine->OnDraw();
					OnPaint();
					//SetWindowTitle(std::to_string(scanline));
					drawed = true;
				}
				if (scanline < lastline)
					drawed = false;

				lastline = scanline;
			}
			
			Engine::pEngine->OnUpdate();
			
		}		
	}
	
	image<Color>* WindowsPlatform::loadImage(std::string path)
	{
		Surface* surface = LoadSurface(myfs::s2w(path));
		if (surface == nullptr) return nullptr;
		if (surface->GetDepth() != 32)
		{
			Surface* newsurface = ChangeDepthSurface(surface, 32);
			delete surface;
			surface = newsurface;
		}
		return new WindowsImage(surface);
	}
	
	void WindowsPlatform::OnPaint()
	{						
		int w = backSurface->GetWidth();
		int h = backSurface->GetHeight();
		int pw = Engine::pEngine->pixelWidth;
		int ph = Engine::pEngine->pixelHeight;
		HDC hdc = GetDC(hWnd);
		if ((pw == 1) && (ph == 1))
			BitBlt(hdc, 0, 0, w, h, backSurface->GetDC(), 0, 0, SRCCOPY);
		else
			StretchBlt(hdc, 0, 0, w * pw, h * ph, backSurface->GetDC(), 0, 0, w, h, SRCCOPY);
		ReleaseDC(hWnd,hdc);		
	}

	void WindowsPlatform::ClearBackground(Color c)
	{	
		backSurface->Clear(c);
	}
	inline int WindowsPlatform::GetScanLine()
	{
		DWORD scanLine;
		HRESULT hr = ((LPDIRECTDRAW)g_pDD)->GetScanLine(&scanLine);
		if (hr == S_OK)
			return (int)scanLine;
		else
			return -1;
	}

    bool WindowsPlatform::AddWindow(int width,int height, int pixelWidth, int pixelHeight, bool fullScreen)
    {
		if (hWnd != nullptr)
		{
			debug << "Window already added. \n";
			return false;
		}		
		WNDCLASS wc;
		wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
		wc.hCursor = LoadCursor(NULL, IDC_ARROW);
		wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
		wc.hInstance = GetModuleHandle(nullptr);	//exe Instance handle
		wc.lpfnWndProc = WindowEvent;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = 0;
		wc.lpszMenuName = nullptr;
		wc.hbrBackground = nullptr;
		wc.lpszClassName = myT("MyEngine");
		RegisterClass(&wc);

		int x = 0;
		int y = 0;

		// Define window furniture
		DWORD dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE ;
		DWORD dwStyle = WS_CAPTION | WS_SYSMENU | WS_VISIBLE | WS_THICKFRAME;

		HMONITOR hmon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
		MONITORINFO mi = { sizeof(mi) };
		if (!GetMonitorInfo(hmon, &mi)) return false;
		Engine::pEngine->monitorWidth = mi.rcMonitor.right;
		Engine::pEngine->monitorHeight = mi.rcMonitor.bottom;
		
		// Handle Fullscreen
		if (fullScreen)
		{
			dwExStyle = 0;
			dwStyle = WS_VISIBLE | WS_POPUP;			
			width = mi.rcMonitor.right;
			height = mi.rcMonitor.bottom;
			x = 0;
			y = 0;
			Engine::pEngine->clientWidth = width/pixelWidth;
			Engine::pEngine->clientHeight = height/pixelHeight;
		}

		// Keep client size as requested
		RECT rWndRect = { 0, 0, width, height };
		AdjustWindowRectEx(&rWndRect, dwStyle, FALSE, dwExStyle);
		int w = rWndRect.right - rWndRect.left;
		int h = rWndRect.bottom - rWndRect.top;		
		backSurface = new Surface();
		if (backSurface->Create(width/pixelWidth, height/pixelHeight, 32) == false) return false;
		Engine::pEngine->background = new WindowsImage(backSurface);
		hWnd = CreateWindowEx(dwExStyle, myT("MyEngine"), myT(""), dwStyle,
			x, y, w, h, NULL, NULL, GetModuleHandle(nullptr), this);
						
		return true;
    }

	LRESULT CALLBACK WindowsPlatform::WindowEvent(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
	{
		switch (uMsg)
		{
			case WM_MOUSEMOVE:
			{				
				int pw = Engine::pEngine->pixelWidth;
				int ph = Engine::pEngine->pixelHeight;
				float mouseX = (float)(lParam & 0xFFFF) / pw;
				float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;
				Engine::pEngine->mouseX = mouseX;
				Engine::pEngine->mouseY = mouseY;
				Engine::pEngine->OnMouse(MouseEvent::Mouse_Move,mouseX, mouseY );
				return 0;				
			}
			//case WM_SIZE:       ptrPGE->olc_UpdateWindowSize(lParam & 0xFFFF, (lParam >> 16) & 0xFFFF);	return 0;
			//case WM_MOUSEWHEEL:	ptrPGE->olc_UpdateMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));           return 0;
			//case WM_MOUSELEAVE: ptrPGE->olc_UpdateMouseFocus(false);                                    return 0;
			//case WM_SETFOCUS:	ptrPGE->olc_UpdateKeyFocus(true);                                       return 0;
			//case WM_KILLFOCUS:	ptrPGE->olc_UpdateKeyFocus(false);                                      return 0;
			case WM_KEYDOWN:	Engine::pEngine->UpdateKeyState(mapKeys[wParam], true);						return 0;
			case WM_KEYUP:		Engine::pEngine->UpdateKeyState(mapKeys[wParam], false);                     return 0;
			case WM_LBUTTONDOWN:
			{
				Engine::pEngine->mousePressed = true;
				int pw = Engine::pEngine->pixelWidth;
				int ph = Engine::pEngine->pixelHeight;
				float mouseX = (float)(lParam & 0xFFFF) / pw;
				float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;
				Engine::pEngine->mouseX = mouseX;
				Engine::pEngine->mouseY = mouseY;
				Engine::pEngine->OnMouse(MouseEvent::Mouse_LBPressed, mouseX, mouseY);
				return 0;
			}
			case WM_LBUTTONUP:
			{
				Engine::pEngine->mousePressed = false;
				int pw = Engine::pEngine->pixelWidth;
				int ph = Engine::pEngine->pixelHeight;
				float mouseX = (float)(lParam & 0xFFFF) / pw;
				float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;
				Engine::pEngine->mouseX = mouseX;
				Engine::pEngine->mouseY = mouseY;
				Engine::pEngine->OnMouse(MouseEvent::Mouse_LBReleased, mouseX, mouseY);
				return 0;
			}
			//case WM_RBUTTONDOWN:ptrPGE->olc_UpdateMouseState(1, true);                                  return 0;
			//case WM_RBUTTONUP:	ptrPGE->olc_UpdateMouseState(1, false);                                 return 0;
			//case WM_MBUTTONDOWN:ptrPGE->olc_UpdateMouseState(2, true);                                  return 0;
			//case WM_MBUTTONUP:	ptrPGE->olc_UpdateMouseState(2, false);                                 return 0;
			case WM_CLOSE:		Engine::baThreadActive = false; PostMessage(hWnd, WM_DESTROY, 0, 0);        return 0;
			case WM_DESTROY:	PostQuitMessage(0); DestroyWindow(hWnd);								return 0;
			//case WM_TIMER:		_OnPaint();/* InvalidateRect(hWnd, nullptr, false);*/					return 0;
			case WM_PAINT:		return 0;
			case WM_ERASEBKGND: return 0;
		}
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}

	void WindowsPlatform::StartSystemEventLoop()
	{
		if (hWnd == nullptr) return;
		MSG msg;
		while (GetMessage(&msg, NULL, 0, 0) > 0)
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
			Engine::pEngine->OnIdle();
		}
		debug << "End of loop";
	}

	
	void WindowsPlatform::SetFPS(int fps)
	{
		
	}

	Color* WindowsPlatform::GetLinePointer(int nLine)
	{
		return (Color*)backSurface->GetLinePointer(nLine);
	}

	void WindowsPlatform::SetWindowTitle(std::string title)
	{				
		SetWindowText(hWnd, ConvertS2W(title).c_str());
	}

	void WindowsPlatform::MapKeys()
	{
		// Create Keyboard Mapping
		mapKeys[0x00] = Key::NONE;
		mapKeys[0x41] = Key::A; mapKeys[0x42] = Key::B; mapKeys[0x43] = Key::C; mapKeys[0x44] = Key::D; mapKeys[0x45] = Key::E;
		mapKeys[0x46] = Key::F; mapKeys[0x47] = Key::G; mapKeys[0x48] = Key::H; mapKeys[0x49] = Key::I; mapKeys[0x4A] = Key::J;
		mapKeys[0x4B] = Key::K; mapKeys[0x4C] = Key::L; mapKeys[0x4D] = Key::M; mapKeys[0x4E] = Key::N; mapKeys[0x4F] = Key::O;
		mapKeys[0x50] = Key::P; mapKeys[0x51] = Key::Q; mapKeys[0x52] = Key::R; mapKeys[0x53] = Key::S; mapKeys[0x54] = Key::T;
		mapKeys[0x55] = Key::U; mapKeys[0x56] = Key::V; mapKeys[0x57] = Key::W; mapKeys[0x58] = Key::X; mapKeys[0x59] = Key::Y;
		mapKeys[0x5A] = Key::Z;

		mapKeys[VK_F1] = Key::F1; mapKeys[VK_F2] = Key::F2; mapKeys[VK_F3] = Key::F3; mapKeys[VK_F4] = Key::F4;
		mapKeys[VK_F5] = Key::F5; mapKeys[VK_F6] = Key::F6; mapKeys[VK_F7] = Key::F7; mapKeys[VK_F8] = Key::F8;
		mapKeys[VK_F9] = Key::F9; mapKeys[VK_F10] = Key::F10; mapKeys[VK_F11] = Key::F11; mapKeys[VK_F12] = Key::F12;

		mapKeys[VK_DOWN] = Key::DOWN; mapKeys[VK_LEFT] = Key::LEFT; mapKeys[VK_RIGHT] = Key::RIGHT; mapKeys[VK_UP] = Key::UP;
		mapKeys[VK_RETURN] = Key::ENTER; //mapKeys[VK_RETURN] = Key::RETURN;

		mapKeys[VK_BACK] = Key::BACK; mapKeys[VK_ESCAPE] = Key::ESCAPE; mapKeys[VK_RETURN] = Key::ENTER; mapKeys[VK_PAUSE] = Key::PAUSE;
		mapKeys[VK_SCROLL] = Key::SCROLL; mapKeys[VK_TAB] = Key::TAB; mapKeys[VK_DELETE] = Key::DEL; mapKeys[VK_HOME] = Key::HOME;
		mapKeys[VK_END] = Key::END; mapKeys[VK_PRIOR] = Key::PGUP; mapKeys[VK_NEXT] = Key::PGDN; mapKeys[VK_INSERT] = Key::INS;
		mapKeys[VK_SHIFT] = Key::SHIFT; mapKeys[VK_CONTROL] = Key::CTRL;
		mapKeys[VK_SPACE] = Key::SPACE;

		mapKeys[0x30] = Key::K0; mapKeys[0x31] = Key::K1; mapKeys[0x32] = Key::K2; mapKeys[0x33] = Key::K3; mapKeys[0x34] = Key::K4;
		mapKeys[0x35] = Key::K5; mapKeys[0x36] = Key::K6; mapKeys[0x37] = Key::K7; mapKeys[0x38] = Key::K8; mapKeys[0x39] = Key::K9;

		mapKeys[VK_NUMPAD0] = Key::NP0; mapKeys[VK_NUMPAD1] = Key::NP1; mapKeys[VK_NUMPAD2] = Key::NP2; mapKeys[VK_NUMPAD3] = Key::NP3; mapKeys[VK_NUMPAD4] = Key::NP4;
		mapKeys[VK_NUMPAD5] = Key::NP5; mapKeys[VK_NUMPAD6] = Key::NP6; mapKeys[VK_NUMPAD7] = Key::NP7; mapKeys[VK_NUMPAD8] = Key::NP8; mapKeys[VK_NUMPAD9] = Key::NP9;
		mapKeys[VK_MULTIPLY] = Key::NP_MUL; mapKeys[VK_ADD] = Key::NP_ADD; mapKeys[VK_DIVIDE] = Key::NP_DIV; mapKeys[VK_SUBTRACT] = Key::NP_SUB; mapKeys[VK_DECIMAL] = Key::NP_DECIMAL;		
	}
	
	webview* WindowsPlatform::AddWebView(int x, int y, int width, int height)
	{
		windowswebview* view = new windowswebview(hWnd, x, y, width, height);
		return view;
	}
}