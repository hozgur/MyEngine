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
	pixel(T x, T y, My::Color c, T vx = 0, T vy = 0, int type = 0) :x(x), y(y), vx(vx), vy(vy), type(type), color(c) {}
};

