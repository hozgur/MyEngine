#pragma once

namespace My
{
	class webview : public object
	{
	public:
		webview(){}
		virtual ~webview(){}
		virtual bool Navigate(std::string url) = 0;
		virtual bool NavigateContent(std::string htmlContent) = 0;
		virtual bool SetScript(std::string scriptContent) = 0;
		virtual bool PostWebMessage(std::string message) = 0;
	};
}