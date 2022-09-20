#pragma once

class myWebView : public myObject
{
public:
	myWebView(){}
	virtual ~myWebView(){}
	virtual bool Navigate(std::string url) = 0;
	virtual bool NavigateContent(std::string htmlContent) = 0;
	virtual bool SetScript(std::string scriptContent) = 0;
	virtual bool PostWebMessage(std::string message) = 0;
};