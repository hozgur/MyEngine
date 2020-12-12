#pragma once

class CColorLineReader
{
public:
	virtual int GetWidth() = 0;
	virtual int GetHeight() = 0;
	virtual short* ReadLine(int line, int color) = 0;
	
	virtual short Get(int x, int y, int c)
	{
		int xx = mod(x, GetWidth());
		int yy = mod(y, GetHeight());
		return ReadLine(yy, c)[xx];
	}
};

class CColorLineWriter
{
public:
	virtual int GetWidth() = 0;
	virtual int GetHeight() = 0;
	virtual void WriteLine(int line, int color, const short* buffer) = 0;
};
