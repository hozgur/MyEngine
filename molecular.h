#pragma once
#include "core.h"
#include <algorithm>

#define MAXVALUE 32767
namespace Molecular
{
	template<typename T>
	struct pair2
	{
		pair2() :x(0), y(0) {}
		pair2(T x, T y) :x(x), y(y) {}
		T x;
		T y;		
	};
	typedef pair2<float> fpair2;
	template<typename T>
	struct pair3
	{
		pair3() :x(0), y(0), z(0) {}
		pair3(T x, T y, T z) :x(x), y(y),z(z) {}
		T x;
		T y;
		T z;
	};
	typedef pair3<float> fpair3;

	template<typename T>
	std::ostream& operator<<(std::ostream& os, const pair2<T> &p)
	{
		os << "( " << p.x << " , " << p.y << " )";
		return os;
	}
	template<typename T>
	std::ostream& operator<<(std::ostream& os, const pair3<T>& p)
	{
		os << "( " << p.x << " , " << p.y << " , " << p.z << " )";
		return os;
	}
	// Graph : https://www.desmos.com/calculator/qfdiokzaet
	
	struct attraction2
	{
		float pullForce = 3.f;
		float pushForce = 50.f;
		float forceRangeCoeff1 = 8;
		float forceRangeCoeff2 = 1;
		float a = 6;
		float distCoeff = 4.5;
		float dumpCoeff = 0.2f;
		
		float func1(float x)
		{			
			return   - pullForce * exp(-(x - a) * (x - a) / forceRangeCoeff1) + pushForce * exp(-(x - a / 2) * (x - a / 2) / forceRangeCoeff2);
		}

		fpair3 func(int idx, int idy)
		{
			float dx = idx * distCoeff / MAXVALUE;
			float dy = idy * distCoeff / MAXVALUE;
			float dist = sqrt(dx * dx + dy * dy);			
			//if (dist < distCoeff*2)
			{
				float norm_dx = dx / dist;
				float norm_dy = dy / dist;
				float f = func1(dist) / dumpCoeff;				
				return fpair3(f * norm_dx, f * norm_dy, f);
			}
			//else
				//return fpair(0, 0);
			

		}	
	}attTable2;

	struct pixel
	{
		int16_t x;
		int16_t y;
		float vx;
		float vy;
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

		pixel():x(0),y(0),vx(0),vy(0),aggregate(0){}
		pixel(int x,int y,My::Color c,float vx = 0, float vy = 0,int type = 0) :x(x), y(y), vx(vx), vy(vy), type(type),color(c) {}
	};

	struct cell
	{
		My::fastarray<pixel, 50> pixels;
		bool bDrawBorder = false;
		int mouseX;
		int mouseY;
		void add(pixel p)
		{
			pixels.push(p);
		}
		void remove(int index)
		{
			pixels.remove(index);
		}
		void clear() { pixels.clear();}
		const My::Color color()
		{			
			const int size = pixels.size();
			if (size > 0)
			{
				uint16_t red = 0;
				uint16_t green = 0;
				uint16_t blue = 0;
				uint16_t alpha = 0;

				for (int a = 0; a < size; a++)
				{
					red += pixels[a].color.r;
					green += pixels[a].color.g;
					blue += pixels[a].color.b;
					alpha += pixels[a].color.a;
				}
				return My::Color(red / size, green / size, blue / size, alpha / size);
			}
			else
				return My::Color::Black;
			
		}
		void draw(int x, int y, int width, int height)
		{
			ivec2 pos(x * width, y * width);
			foreachPixel([&](int i, pixel& p) {
				int px = x * width + p.x * width / MAXVALUE;
				int py = y * height + p.y * height / MAXVALUE;
				//My::Engine::pEngine->Pixel(px, py) =  p.color;
				My::Engine::pEngine->FillCircle(ivec2(px, py), 3,p.color);
				if (bDrawBorder)
				{										
					int mx = pos.x() + mouseX * (width - 1) / MAXVALUE;
					int my = pos.y() + mouseY * (height - 1) / MAXVALUE;
					//My::Engine::pEngine->DrawLine(ivec2(mx, my), ivec2(px, py), My::Color::Gray);
				}
				});
			if (bDrawBorder)
			{				
				My::Engine::pEngine->DrawRect(pos, ivec2(width, height), My::Color::Gray);
				bDrawBorder = false;				
			}
		}

		template<typename T> void foreachPixel(T&& lambda)
		{
//#pragma omp parallel for
			for (int i = 0; i < pixels.size(); i++)							
				lambda(i,pixels[i]);			
		}
		
	};
	
	struct grid
	{
		// Member Variables
	public:
		bool bDrawLines;
	private:
		int cellCountX;
		int cellCountY;
		int currentCellGroup;
		
		cell* cells[2];
		int borderTableCX;
		int borderTableCY;
		int mX;
		int mY;
		int sX;
		int sY;
		int *borderTableX;
		int *borderTableY;
		int scale;		

	public:
		grid() :cellCountX(0), cellCountY(0), borderTableCX(0), borderTableCY(0), scale(1), bDrawLines(false),
			mX(0), mY(0), sX(0), sY(0),borderTableX(nullptr),borderTableY(nullptr),currentCellGroup(0) { assert(RAND_MAX == 32767); }

		grid(int cellcountX, int cellcountY,int scale = 1)
		{
			create(cellcountX, cellcountY);
			this->scale = scale;
		}

		bool create(int cellcountx, int cellcounty)
		{			
			if (cells[0] != nullptr) return false;
			sX = sY = 0;
			cellCountX = cellcountx;
			cellCountY = cellcounty;
			int bit = 1;			
			while (bit < cellcountx) { bit <<= 1; sX++; }
			borderTableCX = bit;
			bit = 1;
			while (bit < cellcounty) { bit <<= 1; sY++; }
			borderTableCY = bit;			
			cells[0] = new cell[cellCountX * cellCountY];
			cells[1] = new cell[cellCountX * cellCountY];
			currentCellGroup = 0;
			mX = borderTableCX - 1;
			mY = borderTableCY - 1;
			createBorderTable();
			return true;
		}

		void createBorderTable()
		{
			borderTableX = new int[borderTableCX];
			borderTableY = new int[borderTableCY];
			for (int i = 0; i < borderTableCX; i++)
				borderTableX[i] = i % cellCountX;
			for (int i = 0; i < borderTableCY; i++)
				borderTableY[i] = i % cellCountY;
		}
		~grid()
		{
			delete[] cells[0];
			cells[0] = nullptr;
			delete[] cells[1];
			cells[1] = nullptr;
			delete borderTableX;
			delete borderTableY;
		}

		cell& getCell(int x, int y){return cells[currentCellGroup][borderTableY[y & mY] * cellCountX + borderTableX[x & mX]];}
		cell& getCell2(int x, int y) { return cells[(currentCellGroup+1) & 1][borderTableY[y & mY] * cellCountX + borderTableX[x & mX]]; }
		void setPixel(int x, int y,const My::Color &color,float vx = 0,float vy = 0,int type = 0)
		{			
			int s = 1;
			int cx = x / s;
			int cy = y / s;
			cell& c = getCell(cx, cy);
			//c.clear();
			//c.add(pixel(std::rand()/4,std::rand()/4,color));
			c.add(pixel((x-cx*s)*32768/s, (y - cy * s) * 32768/s, color, vx, vy,type));
			
		}

		const My::Color getColor(int x, int y)
		{			
			return getCell(x,y).color();
		}

		template<typename T> void foreachCell(T&& lambda)
		{			
#pragma omp parallel for
			for (int y = 0; y < cellCountY; y++)
			{
				cell* pC = &getCell(0,y);
				for (int x = 0; x < cellCountX; x++)
					lambda(x, y, pC[x]);
			}
		}
		template<typename T> void foreachCell2(T&& lambda)
		{
#pragma omp parallel for
			for (int y = 0; y < cellCountY; y++)
			{
				cell* pC = &getCell2(0, y);
				for (int x = 0; x < cellCountX; x++)
					lambda(x, y, pC[x]);
			}
		}
		// index :
		//09 10 11 12 13
		//24 01 02 03 14
		//23 08 00 04 15
		//22 07 06 05 16
		//21 20 19 18 17

		//  -2 -1  0  1  2
		//  -2 -1  0  1  2
		//  -2 -1  x  1  2
		//  -2 -1  0  1  2
		//  -2 -1  0  1  2
		//------------------
		//  -2 -2 -2 -2 -2
		//  -1 -1 -1 -1 -1 
		//   0  0  y  0  0
		//   1  1  1  1  1
		//   2  2  2  2  2

		#define ROUND_TABLE_SIZE  9
		//                  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
		const int dX[25] = {0,-1, 0, 1, 1, 1, 0,-1,-1,-2,-1, 0, 1, 2, 2, 2, 2, 2, 1, 0,-1,-2,-2,-2,-2 };
		const int dY[25] = {0,-1,-1,-1, 0, 1, 1, 1, 0,-2,-2,-2,-2,-2,-1, 0, 1, 2, 2, 2, 2, 2, 1, 0,-1 };

		void runCell(int cx, int cy)
		{
			cell &c = getCell(cx, cy);
			for(int i1 = 0;i1 < c.pixels.size();i1++)
			{
				pixel p1 = c.pixels[i1];

				float descent = 0.995f;
				if (p1.type == 1)
					descent = 0.99999f;
				float vx = p1.vx * descent;
				float vy = p1.vy * descent;
				int x = p1.x;
				int y = p1.y;
				for (int i = 0; i < ROUND_TABLE_SIZE; i++)
				{					
					cell& c1 = getCell(cx + dX[i], cy + dY[i]);
					for (int i2 = 0; i2 < c1.pixels.size(); i2++)
					{						
						if ((i != 0) || (i1 != i2))
						{
							pixel p2 = c1.pixels[i2];
							int dx = p2.x + dX[i] * (MAXVALUE + 1) - p1.x;
							int dy = p2.y + dY[i] * (MAXVALUE + 1) - p1.y;
							fpair3 dv = attTable2.func(dx, dy);
							vx -= dv.x;
							vy -= dv.y;

							if ((scale > 1) && bDrawLines)
							{
								int x1 = cx * scale + p1.x * scale / (MAXVALUE + 1);
								int y1 = cy * scale + p1.y * scale / (MAXVALUE + 1);
								int x2 = (cx + dX[i]) * scale + p2.x * scale / (MAXVALUE + 1);
								int y2 = (cy + dY[i]) * scale + p2.y * scale / (MAXVALUE + 1);
								if ((dv.x != 0) || (dv.y != 0))
								{
									//My::Engine::pEngine->DrawLine(ivec2(x1, y1), ivec2(x2, y2), My::Color::Gray);
								}
							}
						}
					}					
					if (c1.bDrawBorder)
					{
						int dx = c1.mouseX + dX[i] * (MAXVALUE + 1) - p1.x;
						int dy = c1.mouseY + dY[i] * (MAXVALUE + 1) - p1.y;
						fpair3 dv = attTable2.func(dx, dy);
						vx -= dv.x*1;
						vy -= dv.y*1;
						if ((scale > 1) && bDrawLines)
						{
							int x1 = (int) My::Engine::pEngine->mouseX;
							int y1 = (int) My::Engine::pEngine->mouseY;
							int x2 = cx * scale + p1.x * scale / (MAXVALUE + 1);
							int y2 = cy * scale + p1.y * scale / (MAXVALUE + 1);
							if (dv.z != 0)
							{
								if(dv.z < 0)
									My::Engine::pEngine->DrawLine(ivec2(x1, y1), ivec2(x2, y2), My::Color::Gray);
								else
									My::Engine::pEngine->DrawLine(ivec2(x1, y1), ivec2(x2, y2), My::Color::Red);
							}
								
						}
					}
					
				}

				// Eðer velocity çok fazla olursa birden aktarmak yerine parçalý aktarma denenebilir.
				x += (int) vx ;
				y += (int) vy ;
				pixel p(x & MAXVALUE, y & MAXVALUE, p1.color,vx,vy,p1.type);
				
				int ccx = cx + (x >> 15);
				cell& c1 = getCell2(ccx, cy + (y >> 15));
				c1.add(p);
			}
		}
	
		void Run()
		{
			foreachCell2([this](int x, int y, cell& c) { c.clear(); });
			foreachCell([this](int x, int y, cell& c) { runCell(x, y); });
			currentCellGroup = (currentCellGroup + 1) & 1;
		}

	};	
}