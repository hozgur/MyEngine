#ifdef WINDOWS
#include "my.h"
#include "myimage.h"
#include "Windows\windowscore.h"
#include "Windows\windowsplatform.h"
#include <Windows.h>
namespace My
{
	Surface::Surface()
	{
		OldBmp = nullptr;
		SurfaceDC = nullptr;
		Width = Height = 0;
		Depth = 0;
		dpiX = 96;
		dpiY = 96;
		Bitmap = nullptr;
		pData = nullptr;
		BitmapInfo = nullptr;
	}

	Surface::~Surface()
	{
		Destroy();
	}

	void Surface::Destroy()
	{
		if ((OldBmp != nullptr) && (SurfaceDC != nullptr))
		{
			SelectObject(SurfaceDC, OldBmp);
			OldBmp = nullptr;
		}
		if (Bitmap != nullptr) DeleteObject(Bitmap);
		Width = 0;
		Height = 0;
		Bitmap = nullptr;
		pData = nullptr;
		Depth = 0;
		delete[] BitmapInfo;
		BitmapInfo = nullptr;
	}
	bool Surface::Create(int width, int height, int depth, bool flip, COLORREF* palette, int palettesize)
	{
		if (width <= 0) return false;
		//if (height<=0) return false;
		if ((depth != 32) && (depth != 24) && (depth != 8) && (depth != 4) && (depth != 1)) return false;
		if (SurfaceDC == nullptr)
		{
			SurfaceDC = CreateCompatibleDC(nullptr);
			OldBmp = nullptr;
			if (SurfaceDC == nullptr)
			{
				Debug() << "Error on CreateSurfaceDC";
				return false;
			}
		}
		if (IsValid())
		{
			if ((Width != width) || (Height != height) || (Depth != depth))
			{
				if (OldBmp != NULL)
				{
					SelectObject(SurfaceDC, OldBmp);
					OldBmp = NULL;
				}
				Destroy();
				CreateBitmap(width, height, depth, 72, 72, flip, palette, palettesize);
				if (!IsValid())
				{
					debug << "Error on recreate SurfaceBitmap";
					return false;
				}
			}
		}
		else
		{
			CreateBitmap(width, height, depth, 72, 72, flip, palette, palettesize);
			if (!IsValid())
			{
				debug << "Error on create SurfaceBitmap2";
				return false;
			}
		}

		if (OldBmp == NULL)
			OldBmp = (HBITMAP)SelectObject(SurfaceDC, Bitmap);
		Width = width;
		Height = height;
		return true;
	}
#define WIDTHBYTES(bits)    (((bits) + 31) / 32 * 4)
	bool Surface::CreateBitmap(int width, int height, int depth, float dpix, float dpiy, bool flip, COLORREF* palette, int palettesize)
	{
		if (Bitmap != NULL)
			Destroy();

		Stride = WIDTHBYTES(depth * width);
		BITMAPINFOHEADER BmpInfoHeader;
		BmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
		BmpInfoHeader.biBitCount = (byte)depth;
		BmpInfoHeader.biClrUsed = 0;
		BmpInfoHeader.biCompression = BI_RGB;
		BmpInfoHeader.biHeight = flip ? height : -height;
		BmpInfoHeader.biPlanes = 1;
		BmpInfoHeader.biWidth = width;
		BmpInfoHeader.biXPelsPerMeter = (int)(dpix * 254);
		BmpInfoHeader.biYPelsPerMeter = (int)(dpiy * 254);

		if (depth == 1)
		{
			BitmapInfo = (LPBITMAPINFO) new byte[sizeof(BmpInfoHeader) + 2 * sizeof(RGBQUAD)];
			if (BitmapInfo == NULL)
			{
				Depth = 0;
				Width = 0;
				Height = 0;
				pData = nullptr;
				return false;
			}
			BitmapInfo->bmiHeader = BmpInfoHeader;
			BitmapInfo->bmiColors[0].rgbRed = 0;
			BitmapInfo->bmiColors[0].rgbGreen = 0;
			BitmapInfo->bmiColors[0].rgbBlue = 0;
			BitmapInfo->bmiColors[0].rgbReserved = 0;
			BitmapInfo->bmiColors[1].rgbRed = 255;
			BitmapInfo->bmiColors[1].rgbGreen = 255;
			BitmapInfo->bmiColors[1].rgbBlue = 255;
			BitmapInfo->bmiColors[1].rgbReserved = 0;

		}
		else
			if ((depth == 8) || (depth == 4))
			{
				int psize = 1 << depth;
				if ((palettesize == 0) || (palette == NULL))
				{
					BitmapInfo = (LPBITMAPINFO) new byte[sizeof(BmpInfoHeader) + psize * sizeof(RGBQUAD)];
					if (BitmapInfo == nullptr)
					{
						Destroy();
						return false;
					}
					for (int a = 0; a < psize; a++)	// Create GrayScale Palette
					{
						BitmapInfo->bmiColors[a].rgbRed = a;
						BitmapInfo->bmiColors[a].rgbGreen = a;
						BitmapInfo->bmiColors[a].rgbBlue = a;
						BitmapInfo->bmiColors[a].rgbReserved = 0;
					}
					BitmapInfo->bmiHeader = BmpInfoHeader;
					BitmapInfo->bmiHeader.biClrUsed = 0;
				}
				else
				{
					BitmapInfo = (LPBITMAPINFO) new byte[sizeof(BmpInfoHeader) + palettesize];
					if (BitmapInfo == nullptr)
					{
						Destroy();
						return false;
					}
					memcpy(BitmapInfo->bmiColors, palette, palettesize);
					BitmapInfo->bmiHeader = BmpInfoHeader;
					BitmapInfo->bmiHeader.biClrUsed = 0;//palettesize/4;
				}
			}
			else
			{
				BitmapInfo = (LPBITMAPINFO) new byte[sizeof(BmpInfoHeader)];
				BitmapInfo->bmiHeader = BmpInfoHeader;
			}


		Bitmap = CreateDIBSection(NULL, BitmapInfo, DIB_RGB_COLORS, (void**)&pData, nullptr, 0);

		if (Bitmap == NULL)
		{
			// Bu bölüme CreateFileMapping konacak ama önce getbmppointer fonksiyonunun değişmesi gerekiyor.
			//Bitmap = CreateDIBSection(NULL,BitmapInfo,DIB_RGB_COLORS,(void**)&Pointer,NULL,0);
			if (Bitmap == NULL)
			{
				Destroy();
				return false;
			}
		}

		ULONGLONG p = (ULONGLONG)pData;
		if ((p & 0xf) != 0)
		{
			Debug() << "16 Byte Alignment Problem for SSE2 at Surface Pointer is = %X" << pData;
			return false;
		}
		Depth = depth;
		Width = width;
		Height = height;
		return true;
	}


	bool Surface::IsValid()
	{
		return ((SurfaceDC != nullptr) && (Bitmap != nullptr));
	}

	void Surface::SetDPI(float dpix, float dpiy)
	{
		if ((dpix == 0) || (dpiy == 0)) return;
		dpiX = dpix;
		dpiY = dpiy;
		BitmapInfo->bmiHeader.biXPelsPerMeter = (int)(dpiX * 10000 / 254);
		BitmapInfo->bmiHeader.biYPelsPerMeter = (int)(dpiY * 10000 / 254);
	}

	void Surface::Clear(Color c)
	{
		if (!IsValid()) return;		
		if (Depth == 32)
		{		
			Color* p = (Color*)GetLinePointer(0);
			long size = Stride * Height/4;
			for (int a = 0; a < size; a++)								
				p[a] = c;
		}
		else
		{
			if (Depth == 24)
			{
				for (int y = 0; y < Height; y++)
				{
					byte* p = (byte*)GetLinePointer(y);
					for (int x = 0; x < Width; x++)
					{
						*p++ = c.b;
						*p++ = c.g;
						*p++ = c.r;
					}
				}
			}
			else
				if (Depth == 8)
				{										
					memset(GetLinePointer(0), Height * Stride, c.GetGrayTone());
				}				
		}
	}

	Surface* LoadSurface(Gdiplus::Bitmap& image)
	{
		Gdiplus::Status a = image.GetLastStatus();
		if (a != Gdiplus::Ok)
		{
			debug << "Error on Loading Surface " << GetLastError();
			return nullptr;
		}

		int iw = image.GetWidth();
		int ih = image.GetHeight();
		float rx = image.GetHorizontalResolution();
		float ry = image.GetVerticalResolution();
		Surface* surface = new Surface();
		if (surface == nullptr) return nullptr;
		Gdiplus::BitmapData* bitmapdata = new Gdiplus::BitmapData;
		Gdiplus::Rect rect(0, 0, iw, ih);
		int PixelFormat = image.GetPixelFormat();
		bool bConvertColorspace = false;
		if (image.GetPixelFormat() == PixelFormat1bppIndexed)
		{
			surface->Create(iw, ih, 1);
		}
		else
			if (image.GetPixelFormat() == PixelFormat8bppIndexed)
			{
				UINT size = image.GetPaletteSize();
				Gdiplus::ColorPalette* palette = (Gdiplus::ColorPalette*)malloc(size);
				image.GetPalette(palette, size);
				surface->Create(iw, ih, 8, FALSE, (COLORREF*)palette->Entries, size - 8);
				free(palette);
			}
			else
				if (image.GetPixelFormat() == PixelFormat4bppIndexed)
				{
					UINT size = image.GetPaletteSize();
					Gdiplus::ColorPalette* palette = (Gdiplus::ColorPalette*)malloc(size);
					image.GetPalette(palette, size);
					surface->Create(iw, ih, 4, FALSE, (COLORREF*)palette->Entries, size - 8);
					free(palette);
				}
				else
					if (image.GetPixelFormat() == PixelFormat24bppRGB)
					{
						if (!surface->Create(iw, ih, 24))
						{
							delete surface;
							delete bitmapdata;
							return nullptr;
						}
					}
					else
						if (image.GetPixelFormat() == PixelFormat32bppARGB)
						{
							if (!surface->Create(iw, ih, 32))
							{
								delete surface;
								delete bitmapdata;
								return nullptr;
							}
						}
						else
							if (image.GetPixelFormat() == PixelFormat32bppCMYK)
							{
								if (!surface->Create(iw, ih, 32))
								{
									delete surface;
									delete bitmapdata;
									return nullptr;
								}
								//bConvertColorspace = true;
							}
							else
							{
								delete surface;
								delete bitmapdata;
								return nullptr;
							}

		surface->SetDPI(rx, ry);
		int hinterval = ih;
		if (hinterval > 1000) hinterval = 1000;
		int y = 0;
		while (y < ih)
		{
			if ((y + hinterval) > ih) hinterval = ih - y;
			rect.Y = y;
			rect.Height = hinterval;
			if (bConvertColorspace)
			{
				Gdiplus::Graphics g(surface->GetDC());
				g.DrawImage(&image, rect);
			}
			else
			{
				Gdiplus::Status stat = image.LockBits(&rect, Gdiplus::ImageLockModeRead, image.GetPixelFormat(), bitmapdata);
				if (stat != Gdiplus::Ok)
				{
					delete surface;
					delete bitmapdata;
					return nullptr;
				}
				byte* spixels = (byte*)bitmapdata->Scan0;
				long wb = surface->GetStride();
				byte* dpixels = (byte*)surface->GetLinePointer(y);
				UINT size = wb * hinterval;
				CopyMemory(dpixels, spixels, size);
				image.UnlockBits(bitmapdata);
			}

			y += hinterval;
		}

		delete bitmapdata;
		return surface;
	}

	Surface* ChangeDepthSurface(Surface* surface,int depth)
	{
				
		if (surface->GetDepth() == depth) return surface;

		int width = surface->GetWidth();
		int height = surface->GetHeight();
		float dpix = surface->GetDpiX();
		float dpiy = surface->GetDpiY();


		Surface* newsurface = new Surface();
		if (newsurface == NULL)
		{
			debug << "Insufficient Memory Error\n.";
			return nullptr;
		}
		newsurface->Create(width, height, depth);
		newsurface->SetDPI(dpix, dpiy);
		
		BitBlt(newsurface->GetDC(), 0, 0, width, height, surface->GetDC(), 0, 0, SRCCOPY);

		if (newsurface->GetDepth() == 32)	//  yeni surface alpha içeriyor ise alpha'yı tam set et.
		{
			unsigned int* p = (unsigned int*)newsurface->GetLinePointer(0);
			int size = height * newsurface->GetStride()/sizeof(int);
			for (int a = 0; a < size; a++)
			{
				*p = (*p) | 0xFF000000;		// yeni alpha değerlerini 0xFF yap.
				p++;
			}
		}

		return newsurface;
	}
	

	Surface* LoadSurface(std::wstring path)
	{
		Gdiplus::Bitmap image(path.c_str(), TRUE);
		//Gdiplus::Bitmap image(L"c:\\mandala.png", TRUE);
		Surface* surface = LoadSurface(image);
		if (surface == nullptr) return nullptr;
		return surface;
	}

	
}

#endif //WINDOWS