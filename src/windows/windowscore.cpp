#pragma once
#include "core.h"
#include <Windows.h>
#include "..\..\include\windows\windowscore.h"
namespace My
{
	std::wstring ConvertS2W(std::string s)
	{
#ifdef __MINGW32__
		wchar_t* buffer = new wchar_t[s.length() + 1];
		mbstowcs(buffer, s.c_str(), s.length());
		buffer[s.length()] = L'\0';
#else
		int count = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, NULL, 0);
		wchar_t* buffer = new wchar_t[count];
		MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, buffer, count);
#endif
		std::wstring w(buffer);
		delete[] buffer;
		return w;
	}

	std::string ConvertW2S(std::wstring w)
	{
		int count = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, NULL, 0, NULL, NULL);
		char* buffer = new char[count];
		WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, buffer, count, NULL, NULL);

		std::string s(buffer);
		delete[] buffer;
		return s;
	}

	std::wstring GetCWD()
	{
		TCHAR buffer[MAX_PATH] = { 0 };
		GetModuleFileName(NULL, buffer, MAX_PATH);
		std::wstring::size_type pos = std::wstring(buffer).find_last_of(L"\\/");
		return std::wstring(buffer).substr(0, pos);
	}	
}