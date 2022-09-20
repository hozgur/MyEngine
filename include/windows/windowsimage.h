#pragma once
#include "My.h"
#include "Windows\surface.h"
class WindowsImage : public myImage<myColor>
{
public:

	mySurface* surface = nullptr;

	WindowsImage(mySurface* surface)
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

	virtual myColor* readLine(int line) override { return (myColor*)surface->GetLinePointer(line); }

	virtual bool canUseReadforWrite() override { return true; }

	virtual void writeLine(int line, const myColor* data) override { memcpy(surface->GetLinePointer(line), data, getWidth() * sizeof(myColor)); }

	virtual void draw(myImage* img, int x, int y, int dW, int dH, int sX = 0, int sY = 0, int sW = -1, int sH = -1, Interpolation interpolation = Interpolation::interpolation_default) override
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
		}
		else
		{
			//  readLine ile yazmak lazým.
		}
	}
};