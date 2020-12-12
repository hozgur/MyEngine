#pragma once
#include <Windows.h>
#include "mywebview.h"
#include <wrl.h>
#include <wil/com.h>
// include WebView2 header
#include "WebView2.h"
using namespace Microsoft::WRL;


namespace My
{
	class windowswebview : public webview
	{
        wil::com_ptr<ICoreWebView2Controller> webviewController;
        wil::com_ptr<ICoreWebView2> webviewWindow;
        RECT bounds;
        HWND hWnd = nullptr;
        
	public:
		windowswebview(HWND hWnd, int x, int y, int width, int height);
		virtual ~windowswebview();
        bool Create();
        void Close() { webviewController->Close();}
		virtual bool Navigate(std::string url) override
		{
            std::wstring uri(myfs::s2w(url));
			HRESULT hr = webviewWindow->Navigate(uri.c_str());
            if (hr == E_INVALIDARG)
            {
                // An invalid URI was provided.
                if (uri.find(L' ') == std::wstring::npos
                    && uri.find(L'.') != std::wstring::npos)
                {
                    // If it contains a dot and no spaces, try tacking http:// on the front.
                    hr = webviewWindow->Navigate((L"http://" + uri).c_str());
                }
                else
                {
                    // Otherwise treat it as a web search. We aren't bothering to escape
                    // URL syntax characters such as & and #
                    hr = webviewWindow->Navigate((L"https://www.google.com/search?q=" + uri).c_str());
                }
            }
            if (hr != S_OK) {
                debug << "Invalid Arg Error Code hr = " << hr << "\n";
                return false;
            }
            return true;
		}
		virtual bool NavigateContent(std::string htmlContent) override
		{
			HRESULT hr = webviewWindow->NavigateToString(myfs::s2w(htmlContent).c_str());
            if (hr != S_OK)
            {
                debug << "Navigate to Content Error." << hr << "\n";
            }
            return hr == S_OK;
		}

        virtual bool SetScript(std::string scriptContent) override
        {
            HRESULT hr = webviewWindow->AddScriptToExecuteOnDocumentCreated(myfs::s2w(scriptContent).c_str(), nullptr);
            if (hr != S_OK)
            {
                debug << "WebView SetScript Error." << hr << "\n";
                return false;
            }
            return true;
        }

        virtual bool PostWebMessage(std::string message) override
        {
            HRESULT hr = webviewWindow->PostWebMessageAsString(myfs::s2w(message).c_str());
            if (hr != S_OK)
            {
                debug << "WebView Post Message Error." << hr << "\n";
                return false;
            }
            return true;
        }

	};
}