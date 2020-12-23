#pragma once
#include "My.h"
#include "Windows\surface.h"
namespace My
{
	class WindowsImage : public image<Color>
	{
	public:

		Surface* surface = nullptr;

		WindowsImage(Surface* surface)
		{
			this->surface = surface;
		}
		~WindowsImage()
		{
			delete surface;
		}
		// Inherited via image
		virtual int getWidth() const override { return surface->GetWidth(); }

		virtual int getHeight() const override { return surface->GetHeight(); }

		virtual Interleave getInterleave() const override { return Interleave::interleave_interleaved; }

		virtual Color* readLine(int line) const override { return (Color*)surface->GetLinePointer(line); }

		virtual bool canUseReadforWrite() const override { return true; }

		virtual void writeLine(int line, const Color* data, int bytecount) override { memcpy(surface->GetLinePointer(line), data, bytecount); }

		virtual void draw(const image* source, int x, int y, int dW, int dH, int sX = 0, int sY = 0, int sW = -1, int sH = -1, Interpolation interpolation = Interpolation::interpolation_default) const override
		{
			const WindowsImage* sourceImage = dynamic_cast<const WindowsImage*>(source);
			if (sourceImage != nullptr)
			{
				if (sW < 0) sW = dW;
				if (sH < 0) sH = dH;
				BLENDFUNCTION func = { AC_SRC_OVER , 0, 255, AC_SRC_ALPHA };
				HDC dc1 = sourceImage->surface->GetDC();
				HDC dc2 = surface->GetDC();
				::AlphaBlend(dc2, x, y, dW, dH, dc1, sX, sY, sW, sH, func);
			}
			else
			{
				sW = dW;	// ilk sürümde scale yok!!
				sH = dH;
				for (int y = 0; y < dH; y++)
				{

				}
			}
		}
	};
}