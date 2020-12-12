#pragma once
#include "my.h"
#include "ilay_io.h"
#include "mymath.h"
#define MAXVALUE 4095

namespace ilay
{
	class HConvolver : public LineReader
	{
		std::vector<short> kernel;
		LineReader* parent;
		LineBuffer* buffers;
		int Radius;
	public:
		HConvolver(LineReader* parent, int radius, int threadCount = 1) :parent(parent), Radius(radius)
		{
			addString("HConvolver");
			buffers = new LineBuffer(GetWidth(), threadCount);
			makeGaussKernel();
		}
		~HConvolver()
		{
			delete buffers;
		}
		void setKernel(const std::vector<short>& kernel) { this->kernel = kernel; }

		void makeGaussKernel()
		{
			kernel.clear();
			for (int a = -Radius; a <= Radius; a++)
			{
				kernel.push_back((short)(MAXVALUE * My::Math::Gauss2((double)a / Radius)));
			}
		}

		virtual int GetWidth() { return parent->GetWidth(); }
		virtual int GetHeight() { return parent->GetHeight(); }
		virtual short* ReadLine(int line, int color)
		{
			addString("HReadLine");
			int w = GetWidth();
			short* buffer = buffers->Get();
			short* source = parent->ReadLine(line, color);
			short* k = kernel.data();
			for (int x = 0; x < w; x++)
			{
				int val = source[x];
				if (val > 0)
				{
					for (int c = -Radius; c <= Radius; c++)
					{
						if (source[(x + c + w) % w] == 0)
							val -= k[c + Radius];
					}
				}
				if (val < 0) val = 0;
				buffer[x] = (short)(val);
			}
			return buffer;
		}
	};
	class VConvolver : public LineReader
	{
		std::vector<short> kernel;
		LineReader* parent;
		LineBuffer* buffers;

		int Radius;
	public:
		VConvolver(LineReader* parent, int radius, int threadCount = 1) :parent(parent), Radius(radius)
		{
			addString("VConvolver");
			buffers = new LineBuffer(GetWidth(), threadCount);
			makeGaussKernel();

		}
		~VConvolver()
		{
			delete buffers;
		}
		void setKernel(const std::vector<short>& kernel) { this->kernel = kernel; }

		void makeGaussKernel()
		{
			kernel.clear();
			for (int a = -Radius; a <= Radius; a++)
			{
				kernel.push_back((short)(MAXVALUE * My::Math::Gauss2((double)a / Radius)));
			}
		}

		virtual int GetWidth() { return parent->GetWidth(); }
		virtual int GetHeight() { return parent->GetHeight(); }

		virtual short* ReadLine(int line, int color)
		{
			addString("VReadLine");
			int w = GetWidth();
			short* buffer = buffers->Get();
			short* k = kernel.data();
			int kernelSize = 2 * Radius + 1;
			short** lineBuffers = new short* [kernelSize];
			for (int a = 0; a < kernelSize; a++)
			{
				lineBuffers[a] = parent->ReadLine(line + a - Radius, color);
			}
			for (int x = 0; x < w; x++)
			{
				int val = lineBuffers[Radius][x];
				if (val > 0)
				{
					for (int c = 0; c < kernelSize; c++)
					{
						if (lineBuffers[c][x] == 0)
							val -= k[c];
					}
				}
				if (val < 0) val = 0;
				buffer[x] = (short)(val);
			}
			delete lineBuffers;
			return buffer;
		}
	};

	class PrintShrinker : public LineReader
	{
		VConvolver vconvolver;
		HConvolver hconvolver;
	public:
		PrintShrinker(LineReader* fileReader, int radiusX, int radiusY, int bufferCount) :
			vconvolver(fileReader, radiusY, bufferCount), hconvolver(&vconvolver, radiusX, bufferCount){}
		virtual int GetWidth() { return vconvolver.GetWidth(); }
		virtual int GetHeight() { return vconvolver.GetHeight(); }
		virtual short* ReadLine(int line, int color) { return hconvolver.ReadLine(line, color); }
	};
}