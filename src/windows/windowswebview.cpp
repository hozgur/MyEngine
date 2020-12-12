#include "my.h"
#include <windows.h>
#include "Windows/windowswebview.h"
bool My::windowswebview::Create()
{
    CreateCoreWebView2EnvironmentWithOptions(nullptr, nullptr, nullptr,
        Callback<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>(
            [this](HRESULT result, ICoreWebView2Environment* env) -> HRESULT {

                // Create a CoreWebView2Controller and get the associated CoreWebView2 whose parent is the main window hWnd
                env->CreateCoreWebView2Controller(hWnd, Callback<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>(
                    [this](HRESULT result, ICoreWebView2Controller* controller) -> HRESULT {
                        if (controller != nullptr) {
                            webviewController = controller;
                            webviewController->get_CoreWebView2(&webviewWindow);
                        }

                        // Add a few settings for the webview
                        // The demo step is redundant since the values are the default settings
                        ICoreWebView2Settings* Settings;
                        webviewWindow->get_Settings(&Settings);
                        Settings->put_IsScriptEnabled(TRUE);
                        Settings->put_AreDefaultScriptDialogsEnabled(TRUE);
                        Settings->put_IsWebMessageEnabled(TRUE);

                        // Resize WebView to fit the bounds of the parent window

                        //GetClientRect(hWnd, &bounds);
                        webviewController->put_Bounds(bounds);

                        // Schedule an async task to navigate to Bing
                        //webviewWindow->Navigate(L"https://www.bing.com/");
                        std::string url = myfs::path("user/webview/index.html");
                        url = "file://" + url;
                        //webviewWindow->Navigate(myfs::s2w(url).c_str());
                        // Step 4 - Navigation events
                        EventRegistrationToken token1;
                        webviewWindow->add_NavigationStarting(Callback<ICoreWebView2NavigationStartingEventHandler>(
                            [this](ICoreWebView2* webview, ICoreWebView2NavigationStartingEventArgs* args) -> HRESULT {
                                PWSTR uri;
                                args->get_Uri(&uri);
                                std::wstring source(uri);
                                My::Engine::pEngine->OnNavigate(object_id,myfs::w2s(source));
                                //debug << myfs::w2s(source.c_str()) << "\n";
                                CoTaskMemFree(uri);
                                return S_OK;
                            }).Get(), &token1);
                        EventRegistrationToken token2;
                        webviewWindow->add_WebMessageReceived(Callback<ICoreWebView2WebMessageReceivedEventHandler>(
                            [this](ICoreWebView2* webview, ICoreWebView2WebMessageReceivedEventArgs* args) -> HRESULT {
                                PWSTR message;
                                args->TryGetWebMessageAsString(&message);
                                My::Engine::pEngine->OnMessageReceived(object_id, myfs::w2s(message));
                                //debug << myfs::w2s(message) << "\n";
                                //webview->PostWebMessageAsString(message);
                                CoTaskMemFree(message);
                                return S_OK;
                            }).Get(), &token2);

                        // Step 5 - Scripting
                        /*webviewWindow->AddScriptToExecuteOnDocumentCreated(
                            L"window.chrome.webview.addEventListener(\'message\', event => alert(event.data));" \
                            L"window.chrome.webview.postMessage(window.document.URL);",
                            nullptr);*/
                        // Step 6 - Communication between host and web content 
                        My::Engine::pEngine->OnReady(object_id);
                        return S_OK;
                    }).Get());
                return S_OK;
            }).Get());
    return true;
}

My::windowswebview::windowswebview(HWND hWnd, int x, int y, int width, int height)
{
    this->bounds = { x, y, x + width, y + height };
    this->hWnd = hWnd;
    Create();
}

My::windowswebview::~windowswebview()
{
    Close();
}
