#pragma once
#include "Core.h"
namespace My
{
	enum class Interpolation {interpolation_default};
	enum class Interleave {interleave_interleaved, interleave_planar};
	template<typename T>
	class image : public object
	{
	public:
		virtual int getWidth() const = 0;
		virtual int getHeight() const = 0;
		virtual Interleave getInterleave() const = 0;
		virtual T* readLine(int line) const = 0;
		virtual bool canUseReadforWrite() const = 0;
		virtual void writeLine(int line, const T* data, int bytecount) = 0;
		virtual void draw(const image* source, int x, int y, int dW, int dH, int sX = 0, int sY = 0, int sW = -1, int sH = -1, Interpolation interpolation = Interpolation::interpolation_default) const {
			debug << "image draw not supported error.\n";
		}
	};
}
