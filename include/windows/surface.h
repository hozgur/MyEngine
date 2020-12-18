#pragma once
namespace My
{
	class Surface
	{
	private:
		int Width;
		int Height;
		HBITMAP OldBmp;
		HBITMAP Bitmap;
		BITMAPINFO* BitmapInfo;
		float dpiX;
		float dpiY;
		uint32_t Stride;
		uint32_t Depth;
		HDC SurfaceDC;
		byte* pData;

	public:
		Surface();
		virtual ~Surface();
		bool Create(int width, int height, int depth, bool flip = false, COLORREF* palette = nullptr, int palettesize = 0);
		void Destroy();
		bool IsValid();

		HDC GetDC()     { return SurfaceDC; }
		int GetWidth()  { return Width; }
		int GetHeight() { return Height; }
		int GetStride() { return Stride; }
		int GetDepth()  { return Depth; }
		float GetDpiX() { return dpiX; }
		float GetDpiY() { return dpiY; }
		void SetDPI(float dpix, float dpiy);
		BITMAPINFO* GetBmpInfo() { return BitmapInfo; }

		void Clear(Color c);
		void* GetLinePointer(int nLine) { return pData + nLine * Stride; }
	protected:


	private:
		bool CreateBitmap(int width, int height, int depth, float dpix, float dpiy, bool flip = false, COLORREF* palette = NULL, int palettesize = 0);
	};
	Surface* LoadSurface(std::wstring path);
	Surface* ChangeDepthSurface(Surface* surface, int depth);
	
}