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
		virtual int getWidth() override { return surface->GetWidth(); }

		virtual int getHeight() override { return surface->GetHeight(); }

		virtual Interleave getInterleave() override { return Interleave::interleave_interleaved; }

		virtual Color* readLine(int line) override { return (Color*)surface->GetLinePointer(line); }

		virtual bool canUseReadforWrite() override { return true; }

		virtual void writeLine(int line, const Color* data) override { memcpy(surface->GetLinePointer(line), data, getWidth() * sizeof(Color)); }

		virtual void draw(image* img, int x, int y, int dW, int dH, int sX = 0, int sY = 0, int sW = -1, int sH = -1, Interpolation interpolation = Interpolation::interpolation_default) override
		{
			WindowsImage* sourceImage = dynamic_cast<WindowsImage*>(img);
			if (sourceImage != nullptr)
			{
				if (sW < 0) sW = dW;
				if (sH < 0) sH = dH;
				BLENDFUNCTION func = { AC_SRC_OVER , 0, 255, AC_SRC_ALPHA };
				HDC dc1 = sourceImage->surface->GetDC();
				HDC dc2 = surface->GetDC();
				::AlphaBlend(dc2, x, y, dW, dH, dc1, sX, sY, sW, sH, func);
				Color* p = sourceImage->readLine(0);
				debug << p[0].a;
				/*if ((sW == dW) && (sH == dH))
				{			
					

					BitBlt(dc2, x, y, dW, dH,dc1, sX, sY, SRCCOPY);
				}
				else
					StretchBlt(surface->GetDC(), x, y, dW, dH, sourceImage->surface->GetDC(), sX, sY, sW, sH, SRCCOPY);*/
			}
			else
			{
				//  readLine ile yazmak lazým.
			}
		}
	};
}