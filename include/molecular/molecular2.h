#pragma once
#include "core.h"

template<typename T>
struct pixel
{
	T x;
	T y;
	T vx;
	T vy;
	union {
		struct {
			uint8_t	mass;
			uint8_t	type;
			uint8_t	reserved1;
			uint8_t	reserved2;
		};
		uint32_t aggregate;
	};

	My::Color color;

	pixel() :x(0), y(0), vx(0), vy(0), aggregate(0) {}
	pixel(My::Color c, T vx = 0, T vy = 0, T x = 0, T y = 0, int type = 0) :x(x), y(y), vx(vx), vy(vy), type(type), color(c) {}
};

template<typename T>
class cell
{
public:
	My::fastarray<pixel<T>> pixels;
	void add(const pixel& p)
	{
		pixels.push(p);
	}
	void clear()
	{
		pixels.clear();
	}
};

template<typename T>	// T should be integer, short vb. type.
class grid
{
	int width;
	int height;
public:
	grid(int width, int height):width(width),height(height)
	{
		
	}

	int add(double x, double y, pixel<int> p)
	{
		T xc = (T) x;
		T yc = (T) y;
		T xi = x - xc;
		T yi = y - yc;
	}

};