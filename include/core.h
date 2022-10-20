#pragma once
#include <cmath>
#include <cstdint>
#include <string>
#include <iostream>
#include <streambuf>
#include <sstream>
#include <chrono>
#include <vector>
#include <queue>
#include <list>
#include <thread>
#include <atomic>
#include <fstream>
#include <map>
#include <functional>
#include <algorithm>
#include <array>
#include <cstring>
#include <cassert>
#include <filesystem>
#include <locale>
#include <codecvt>
#include <variant>
#include <utility>
#include <3rdParty/json.hpp>
using json = nlohmann::json;

#include "limits.h"
#ifdef WINDOWS
#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#endif

#define CONSOLE_DEFAULT		"\x1B[0m"
#define CONSOLE_RED			"\x1B[31m"
#define CONSOLE_GREEN		"\x1B[32m"
#define CONSOLE_YELLOW		"\x1B[33m"
#define CONSOLE_BLUE		"\x1B[34m"
#define CONSOLE_MAGENTA		"\x1B[35m"
#define CONSOLE_CYAN		"\x1B[36m"
#define CONSOLE_WHITE		"\x1B[37m"
#define CONSOLE_BRED		"\x1B[31;1m"
#define CONSOLE_BGREEN		"\x1B[32;1m"
#define CONSOLE_BYELLOW		"\x1B[33;1m"
#define CONSOLE_BBLUE		"\x1B[34;1m"
#define CONSOLE_BMAGENTA	"\x1B[35;1m"
#define CONSOLE_BCYAN		"\x1B[36;1m"
#define CONSOLE_BWHITE		"\x1B[37;1m"

extern std::stringstream debugStringStream;
extern std::vector<std::function<void(std::string)> > debugHandlers;

std::ostream& Debug();


#define debug Debug()


#if defined(UNICODE) || defined(_UNICODE)
#define myT(s) L##s
#else
#define myT(s) s
#endif
#define UNUSED(x) (void)(x)
constexpr uint8_t  nDefaultAlpha = 0xFF;
constexpr uint32_t nDefaultPixel = (nDefaultAlpha << 24);
inline int mod(int a, int b) { return (a % b + b) % b; }

/// <summary>
/// handle 
/// Object Handle
/// </summary>
typedef std::string myString;

typedef int myHandle;	
const myHandle invalidHandle = -1;
typedef std::variant<int, float, const char*> variant;
typedef uint64_t myTime;

// time
// �nemli not: Windows 10 C++ 20 standard�nda file time ve system time d�n���mlerini tam olarak desteklemedi�i ve arka planda
// kendisi filetime kulland��� i�in. time de�i�ken taban� file_clock �zerinden kuruldu.
// B�ylece sistem saati ile dosyalar�n kay�t zamanlar� tutarl� hale getirildi.
namespace myos {
	inline myTime now() {
		return  std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::file_clock::now().time_since_epoch()).count();
	}
	inline std::chrono::time_point<std::chrono::file_clock> timePoint(myTime t) {
		return  std::chrono::time_point<std::chrono::file_clock>(std::chrono::microseconds(t));
	}
}


class myObject
{
protected:
	myHandle object_id;
public:
	myObject():object_id(invalidHandle){};
	virtual ~myObject() {}
	void SetID(myHandle id) { this->object_id = id; }
	myHandle GetID() { return object_id; }
};

struct myColor
{
	union
	{
		uint32_t n = nDefaultPixel;
		struct { uint8_t b; uint8_t g; uint8_t r; uint8_t a; };
	};
	myColor() { r = 0; g = 0; b = 0; a = nDefaultAlpha; }
	myColor(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = nDefaultAlpha) { n = blue | (green << 8) | (red << 16) | (alpha << 24); }
	myColor(uint32_t p) { n = p; }

	uint8_t GetGrayTone() { return (59 * r + 30 * g + 11 * b) / 100; }
	static myColor Random(int min = 0, int max = 255) { return myColor(min + (rand() % (max - min + 1)), min + (rand() % (max - min + 1)), min + (rand() % (max - min + 1))); }
	enum Colors{White = 0xFFFFFFFF,  Red = 0xFFFF0000, DarkRed = 0xff800000, Blue = 0xFF0000FF, Black = 0xFF000000, Gray = 0xFF808080, Green = 0xFF00FF00};

};
	
struct myStopWatch
{
	void Start()
	{
		t1 = std::chrono::high_resolution_clock::now();
	}
	double Stop()
	{
		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
		time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		return time_span.count();
	}
	double GetDurationMS()
	{
		return time_span.count() * 1000.;	// miliseconds
	}
	double GetDurationS()
	{
		return time_span.count();			// seconds.
	}
	std::string GetDurationStr()
	{
		std::stringstream buffer;
		buffer << "It took me " << GetDurationMS() << " miliseconds." << std::endl;
		return buffer.str();
	}

	std::chrono::high_resolution_clock::time_point t1;
	std::chrono::duration<double> time_span;
};
template<typename T = int,int Size = 4>
struct myFastArray
{
	myFastArray():currentSize(0){}
	int size() { return currentSize; }
	void clear() { currentSize = 0; }
	void push(const T& val) {assert(currentSize < Size); data[currentSize++] = val; }
	void remove(int index) { assert(index < currentSize); data[index] = data[--currentSize]; }
	T& operator[](int index) { assert(index < currentSize); return data[index]; }

private:
	T data[Size];
	int currentSize;
};			

namespace myfs
{
	std::string w2s(std::wstring_view wstring);
	std::wstring s2w(std::string_view string);
	std::string root();
	bool exists(const std::string& filePath);
	bool exists(const std::string& path, const std::string& filename);
	std::string path(std::string directory);
	std::string path(std::string directory, std::string filename);
	std::string getEnv(std::string env);
	myTime lastWriteTime(const std::string& filePath);
}

