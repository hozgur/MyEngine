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
#ifdef WINDOWS
#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#endif

inline std::ostream& Debug()
{
    return std::cout;
}

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

namespace My
{	
	/// <summary>
	/// handle 
	/// Object Handle
	/// </summary>
	typedef int handle;	
	const handle invalidHandle = -1;
	class object
	{
	protected:
		handle object_id;		
	public:
		object():object_id(invalidHandle){};
		virtual ~object() {}
		void SetID(handle id) { this->object_id = id; }
		handle GetID() { return object_id; }
	};
	typedef std::variant<int, float, const char*> variant;
	struct Color
	{
		union
		{
			uint32_t n = nDefaultPixel;
			struct { uint8_t b; uint8_t g; uint8_t r; uint8_t a; };
		};
		Color() { r = 0; g = 0; b = 0; a = nDefaultAlpha; }
		Color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = nDefaultAlpha) { n = blue | (green << 8) | (red << 16) | (alpha << 24); }
		Color(uint32_t p) { n = p; }

		uint8_t GetGrayTone() { return (59 * r + 30 * g + 11 * b) / 100; }
		static Color Random(int min = 0, int max = 255) { return Color(min + (rand() % (max-min+1)), min + (rand() % (max - min + 1)), min + (rand() % (max - min + 1))); }
		enum Colors{White = 0xFFFFFFFF,  Red = 0xFFFF0000, DarkRed = 0xff800000, Blue = 0xFF0000FF, Black = 0xFF000000, Gray = 0xFF808080, Green = 0xFF00FF00};

	};
	
	void sleep(int ms);
	struct StopWatch
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
	struct fastarray
	{
		fastarray():currentSize(0){}
		int size() { return currentSize; }
		void clear() { currentSize = 0; }
		void push(const T& val) {assert(currentSize < Size); data[currentSize++] = val; }
		void remove(int index) { assert(index < currentSize); data[index] = data[--currentSize]; }
		T& operator[](int index) { assert(index < currentSize); return data[index]; }

	private:
		T data[Size];
		int currentSize;
	};			
}

namespace myfs
{
	std::string w2s(std::wstring_view wstring);
	std::wstring s2w(std::string_view string);
	std::string root();
	bool exits(const std::string& path, const std::string& filename);
	std::string path(std::string directory);
	std::string getEnv(std::string env);
}

