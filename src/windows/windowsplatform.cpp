#pragma once
#include "my.h"
#include "windows\windowscore.h"
#include "Windows\windowsplatform.h"
#include <Shlwapi.h>
#include <shellscalingapi.h>
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "Shlwapi.lib")
#pragma comment(lib, "ddraw.lib")
#include "Windows\directdraw.h"
#include "Windows\windowswebview.h"
#include "Windows\windowsimage.h"

std::atomic<bool> WindowsPlatform::ThreadActive = { false };
static std::map<size_t, uint8_t> mapKeys;
WindowsPlatform::WindowsPlatform() :myPlatform()
{
	Gdiplus::GdiplusStartupInput gdiplusstartupinput;
	Gdiplus::GdiplusStartup(&m_gdiplusToken, &gdiplusstartupinput, NULL);
	backSurface = nullptr;
	SetConsoleOutputCP(65001);
	SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
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
	static myStopWatch s;
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
				if (d > 15)
				{
					s.Start();
					myEngine::pEngine->OnDraw();
					OnPaint();
				}
			}
		}
		else
		{
			if (!drawed && (scanline > 700))
			{
				myEngine::pEngine->OnDraw();
				OnPaint();
				//SetWindowTitle(std::to_string(scanline));
				drawed = true;
			}
			if (scanline < lastline)
				drawed = false;

			lastline = scanline;
		}
			
		myEngine::pEngine->OnUpdate();
			
	}		
}
	
myImage<myColor>* WindowsPlatform::loadImage(std::string path)
{
	mySurface* surface = LoadSurface(myfs::s2w(path));
	if (surface == nullptr) return nullptr;
	if (surface->GetDepth() != 32)
	{
		mySurface* newsurface = ChangeDepthSurface(surface, 32);
		delete surface;
		surface = newsurface;
	}
	return new WindowsImage(surface);
}
	
void WindowsPlatform::OnPaint()
{						
	int w = backSurface->GetWidth();
	int h = backSurface->GetHeight();
	int pw = myEngine::pEngine->pixelWidth;
	int ph = myEngine::pEngine->pixelHeight;
	HDC hdc = GetDC(hWnd);
	if ((pw == 1) && (ph == 1))
		BitBlt(hdc, 0, 0, w, h, backSurface->GetDC(), 0, 0, SRCCOPY);
	else
		StretchBlt(hdc, 0, 0, w * pw, h * ph, backSurface->GetDC(), 0, 0, w, h, SRCCOPY);
	ReleaseDC(hWnd,hdc);		
}

void WindowsPlatform::ClearBackground(myColor c)
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

bool WindowsPlatform::AddMainWindow(int width,int height, int pixelWidth, int pixelHeight, bool fullScreen)
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
	myEngine::pEngine->monitorWidth = mi.rcMonitor.right;
	myEngine::pEngine->monitorHeight = mi.rcMonitor.bottom;
		
	// Handle Fullscreen
	if (fullScreen)
	{
		dwExStyle = 0;
		dwStyle = WS_VISIBLE | WS_POPUP;			
		width = mi.rcMonitor.right;
		height = mi.rcMonitor.bottom;
		x = 0;
		y = 0;
		myEngine::pEngine->clientWidth = width/pixelWidth;
		myEngine::pEngine->clientHeight = height/pixelHeight;
		RECT rWndRect = { 0, 0, width, height };
		AdjustWindowRectEx(&rWndRect, dwStyle, FALSE, dwExStyle);
	}
	else
	{
		RECT windowRect;
		windowRect.left = (long)0;
		windowRect.right = (long)width;
		windowRect.top = (long)0;
		windowRect.bottom = (long)height;
		AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);
		width = windowRect.right - windowRect.left;
		height = windowRect.bottom - windowRect.top;
		x = (mi.rcMonitor.right - width) / 2;
		y = (mi.rcMonitor.bottom - height) / 2;
		myEngine::pEngine->clientWidth = width / pixelWidth;
		myEngine::pEngine->clientHeight = height / pixelHeight;
	}

	// Keep client size as requested
	
	int w = myEngine::pEngine->clientWidth;
	int h = myEngine::pEngine->clientHeight;
	backSurface = new mySurface();
	if (backSurface->Create(w, h, 32) == false) return false;
	myEngine::pEngine->background = new WindowsImage(backSurface);
	hWnd = CreateWindowEx(dwExStyle, myT("MyEngine"), myT(""), dwStyle,
		x, y, w, h, NULL, NULL, GetModuleHandle(nullptr), this);
						
	return true;
}

bool WindowsPlatform::DestroyMainWindow()
{
	if (hWnd == nullptr) return false;
	DestroyWindow(hWnd);
	hWnd = nullptr;
	return true;
}

LRESULT CALLBACK WindowsPlatform::WindowEvent(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg)
	{
		case WM_MOUSEMOVE:
		{				
			int pw = myEngine::pEngine->pixelWidth;
			int ph = myEngine::pEngine->pixelHeight;
			float mouseX = (float)(lParam & 0xFFFF) / pw;
			float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;			
			myEngine::pEngine->mouseX = mouseX;
			myEngine::pEngine->mouseY = mouseY;
			myEngine::pEngine->OnMouse(myMouseEvent::Mouse_Move,mouseX, mouseY );
			return 0;				
		}
		case WM_SIZE:       myEngine::pEngine->onSize(lParam & 0xFFFF, (lParam >> 16) & 0xFFFF);	return 0;
		//case WM_MOUSEWHEEL:	ptrPGE->olc_UpdateMouseWheel(GET_WHEEL_DELTA_WPARAM(wParam));           return 0;
		//case WM_MOUSELEAVE: ptrPGE->olc_UpdateMouseFocus(false);                                    return 0;
		//case WM_SETFOCUS:	ptrPGE->olc_UpdateKeyFocus(true);                                       return 0;
		//case WM_KILLFOCUS:	ptrPGE->olc_UpdateKeyFocus(false);                                      return 0;
		case WM_KEYDOWN:	myEngine::pEngine->UpdateKeyState(mapKeys[wParam], true);						return 0;
		case WM_KEYUP:		myEngine::pEngine->UpdateKeyState(mapKeys[wParam], false);                     return 0;
		case WM_LBUTTONDOWN:
		{
			myEngine::pEngine->mousePressed = true;
			int pw = myEngine::pEngine->pixelWidth;
			int ph = myEngine::pEngine->pixelHeight;
			float mouseX = (float)(lParam & 0xFFFF) / pw;
			float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;
			myEngine::pEngine->mouseX = mouseX;
			myEngine::pEngine->mouseY = mouseY;
			myEngine::pEngine->OnMouse(myMouseEvent::Mouse_LBPressed, mouseX, mouseY);
			return 0;
		}
		case WM_LBUTTONUP:
		{
			myEngine::pEngine->mousePressed = false;
			int pw = myEngine::pEngine->pixelWidth;
			int ph = myEngine::pEngine->pixelHeight;
			float mouseX = (float)(lParam & 0xFFFF) / pw;
			float mouseY = (float)((lParam >> 16) & 0xFFFF) / ph;
			myEngine::pEngine->mouseX = mouseX;
			myEngine::pEngine->mouseY = mouseY;
			myEngine::pEngine->OnMouse(myMouseEvent::Mouse_LBReleased, mouseX, mouseY);
			return 0;
		}
		//case WM_RBUTTONDOWN:ptrPGE->olc_UpdateMouseState(1, true);                                  return 0;
		//case WM_RBUTTONUP:	ptrPGE->olc_UpdateMouseState(1, false);                                 return 0;
		//case WM_MBUTTONDOWN:ptrPGE->olc_UpdateMouseState(2, true);                                  return 0;
		//case WM_MBUTTONUP:	ptrPGE->olc_UpdateMouseState(2, false);                                 return 0;
		case WM_CLOSE:		myEngine::baThreadActive = false; PostMessage(hWnd, WM_DESTROY, 0, 0);        return 0;
		case WM_DESTROY:	if(myEngine::pEngine->keepAliveonDestroyWindow == false) PostQuitMessage(0); if(hWnd) DestroyWindow(hWnd); return 0;
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
		myEngine::pEngine->onIdle();
	}
	debug << "End of loop";
}

	
void WindowsPlatform::SetFPS(int fps)
{
		
}

myColor* WindowsPlatform::GetLinePointer(int nLine)
{
	return (myColor*)backSurface->GetLinePointer(nLine);
}

void WindowsPlatform::SetWindowTitle(std::string title)
{				
	SetWindowText(hWnd, myfs::s2w(title).c_str());
}

void WindowsPlatform::MapKeys()
{
	// Create Keyboard Mapping
	mapKeys[0x00] = myKey::NONE;
	mapKeys[0x41] = myKey::A; mapKeys[0x42] = myKey::B; mapKeys[0x43] = myKey::C; mapKeys[0x44] = myKey::D; mapKeys[0x45] = myKey::E;
	mapKeys[0x46] = myKey::F; mapKeys[0x47] = myKey::G; mapKeys[0x48] = myKey::H; mapKeys[0x49] = myKey::I; mapKeys[0x4A] = myKey::J;
	mapKeys[0x4B] = myKey::K; mapKeys[0x4C] = myKey::L; mapKeys[0x4D] = myKey::M; mapKeys[0x4E] = myKey::N; mapKeys[0x4F] = myKey::O;
	mapKeys[0x50] = myKey::P; mapKeys[0x51] = myKey::Q; mapKeys[0x52] = myKey::R; mapKeys[0x53] = myKey::S; mapKeys[0x54] = myKey::T;
	mapKeys[0x55] = myKey::U; mapKeys[0x56] = myKey::V; mapKeys[0x57] = myKey::W; mapKeys[0x58] = myKey::X; mapKeys[0x59] = myKey::Y;
	mapKeys[0x5A] = myKey::Z;

	mapKeys[VK_F1] = myKey::F1; mapKeys[VK_F2] = myKey::F2; mapKeys[VK_F3] = myKey::F3; mapKeys[VK_F4] = myKey::F4;
	mapKeys[VK_F5] = myKey::F5; mapKeys[VK_F6] = myKey::F6; mapKeys[VK_F7] = myKey::F7; mapKeys[VK_F8] = myKey::F8;
	mapKeys[VK_F9] = myKey::F9; mapKeys[VK_F10] = myKey::F10; mapKeys[VK_F11] = myKey::F11; mapKeys[VK_F12] = myKey::F12;

	mapKeys[VK_DOWN] = myKey::DOWN; mapKeys[VK_LEFT] = myKey::LEFT; mapKeys[VK_RIGHT] = myKey::RIGHT; mapKeys[VK_UP] = myKey::UP;
	mapKeys[VK_RETURN] = myKey::ENTER; //mapKeys[VK_RETURN] = myKey::RETURN;

	mapKeys[VK_BACK] = myKey::BACK; mapKeys[VK_ESCAPE] = myKey::ESCAPE; mapKeys[VK_RETURN] = myKey::ENTER; mapKeys[VK_PAUSE] = myKey::PAUSE;
	mapKeys[VK_SCROLL] = myKey::SCROLL; mapKeys[VK_TAB] = myKey::TAB; mapKeys[VK_DELETE] = myKey::DEL; mapKeys[VK_HOME] = myKey::HOME;
	mapKeys[VK_END] = myKey::END; mapKeys[VK_PRIOR] = myKey::PGUP; mapKeys[VK_NEXT] = myKey::PGDN; mapKeys[VK_INSERT] = myKey::INS;
	mapKeys[VK_SHIFT] = myKey::SHIFT; mapKeys[VK_CONTROL] = myKey::CTRL;
	mapKeys[VK_SPACE] = myKey::SPACE;

	mapKeys[0x30] = myKey::K0; mapKeys[0x31] = myKey::K1; mapKeys[0x32] = myKey::K2; mapKeys[0x33] = myKey::K3; mapKeys[0x34] = myKey::K4;
	mapKeys[0x35] = myKey::K5; mapKeys[0x36] = myKey::K6; mapKeys[0x37] = myKey::K7; mapKeys[0x38] = myKey::K8; mapKeys[0x39] = myKey::K9;

	mapKeys[VK_NUMPAD0] = myKey::NP0; mapKeys[VK_NUMPAD1] = myKey::NP1; mapKeys[VK_NUMPAD2] = myKey::NP2; mapKeys[VK_NUMPAD3] = myKey::NP3; mapKeys[VK_NUMPAD4] = myKey::NP4;
	mapKeys[VK_NUMPAD5] = myKey::NP5; mapKeys[VK_NUMPAD6] = myKey::NP6; mapKeys[VK_NUMPAD7] = myKey::NP7; mapKeys[VK_NUMPAD8] = myKey::NP8; mapKeys[VK_NUMPAD9] = myKey::NP9;
	mapKeys[VK_MULTIPLY] = myKey::NP_MUL; mapKeys[VK_ADD] = myKey::NP_ADD; mapKeys[VK_DIVIDE] = myKey::NP_DIV; mapKeys[VK_SUBTRACT] = myKey::NP_SUB; mapKeys[VK_DECIMAL] = myKey::NP_DECIMAL;		
}
	
myWebView* WindowsPlatform::AddWebView(int x, int y, int width, int height, myAnchor anchor)
{
	windowswebview* view = new windowswebview(hWnd, x, y, width, height,anchor);
	return view;
}
