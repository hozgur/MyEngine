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
	void add(const pixel<T>& p)
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
	cell<T>* cells1 = nullptr;
	cell<T>* cells2 = nullptr;
	cell<T>* currentCell = nullptr;
public:
	grid(int width, int height):width(width),height(height)
	{
		int size = width * height;
		cells1 = new cell<T>[size];
		cells2 = new cell<T>[size];
		currentCell = cells1;
	}
	~grid()
	{
		delete cells1;
		delete cells2;
	}
	void add(double x, double y, pixel<int> p)
	{
		T xc = (T) x;
		T yc = (T) y;		
		p.x = (T)(x - xc);
		p.y = (T)(y - yc);
		getCell(xc, yc).add(p);
	}


	// Cell functions

	cell<T>& getCell(int x, int y)
	{
		return currentCell[y * width + x];
	}
};