#pragma once
#include "core.h"
std::wstring ConvertS2W(std::string_view s);
std::string ConvertW2S(std::wstring_view w);
std::wstring GetCWD();	
