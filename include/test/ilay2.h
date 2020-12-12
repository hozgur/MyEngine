#pragma once
#include "my.h"
#include <fstream>
#include "windows\windowscore.h"
#include "windows\windowsplatform.h"

#define MAXVALUE 4095
#include "ilay_imageio.h.h"
#include "mymath.h"

#define QUEUESIZE 20
class CBufferQueue
{
	short** buffers;
	int bufferSize;
	int queueSize;
	int currentQ;
public:
	CBufferQueue(int buffersize, int queuesize): bufferSize(buffersize), queueSize(queuesize)
	{
		buffers = new short* [queuesize];
		for (int a = 0; a < queuesize; a++)
			buffers[a] = new short[buffersize];
		currentQ = 0;
	}

	~CBufferQueue()
	{		
		for (int a = 0; a < queueSize; a++)
			delete buffers[a];
		delete buffers;
	}

	short* Get()
	{
		short* buffer = buffers[currentQ++];
		if (currentQ == queueSize) currentQ = 0;
		return buffer;
	}
};




class CDataFileWriter : public CColorLineWriter
{
	int Width;
	int Height;
	int channelCount;
	std::ofstream file;	
	const int headerSize = 3 * sizeof(int);
public:
	CDataFileWriter(std::string filePath, int width, int height, int channelcount) :Width(width), Height(height), channelCount(channelcount)
	{
		file.open(filePath, std::ios::out | std::ios::binary);
		file.write((char*)&width, sizeof(int));
		file.write((char*)&height, sizeof(int));
		file.write((char*)&channelcount, sizeof(int));
	}

	~CDataFileWriter()
	{
		file.close();
	}

	void Close() { file.close();}

	virtual int GetWidth() { return Width; }
	virtual int GetHeight() { return Height; }
	virtual void WriteLine(int line, int color, const short* buffer)
	{	
		uint64_t pos = ((uint64_t)line *channelCount + color) * Width * sizeof(short) + headerSize;
		file.seekp(pos);
		file.write((char*)buffer, Width * sizeof(short));	
	}
};




bool writetestfile(std::string inputFile, std::string outputFile)
{
	My::StopWatch s;
	s.Start();
	My::Surface *surface = My::LoadSurface(myfs::s2w(inputFile));
	s.Stop();
	debug << "Image File Read Time = " << s.GetDurationS() << " seconds.\n";
	if (surface == nullptr)
	{
		debug << "error on load!";
		return false;
	}
	int scaleshift = 0;
	int w = surface->GetWidth() << scaleshift;
	int h = surface->GetHeight() << scaleshift;
	int d = surface->GetDepth();
	debug << "Width = " << w << " Height = " << h << " Depth = " << d << "\n";
	if (d != 8)
	{
		debug << "Image must be 1 (8bit) channel.\n";
		return false;
	}
	CDataFileWriter writer(outputFile, w, h, 1);
	std::vector<short> buffer(w);
	s.Start();
	for (int y = 0; y < h; y++)
	{
		uint8_t *p = ( uint8_t* ) surface->GetLinePointer(y >> scaleshift);
		for (int c = 0; c < 1; c++)
		{
			for (int x = 0; x < w; x++)
				buffer[x] = 4095 - p[x >> scaleshift] * 4095 / 255;
			writer.WriteLine(y, c, buffer.data());
		}			
	}
	s.Stop();
	debug << "Data File Write Time = " << s.GetDurationS() << " seconds.\n";
	return true;
}
class CDataFileReader : public CColorLineReader
{
protected:
	int Width;
	int Height;
	int channelCount;
	std::ifstream file;
	const int headerSize = 3 * sizeof(int);
	CBufferQueue* buffers;
public:
	CDataFileReader(std::string filePath)
	{
		file.open(filePath, std::ios::in | std::ios::binary);
		file.read((char*)&Width, sizeof(int));
		file.read((char*)&Height, sizeof(int));
		file.read((char*)&channelCount, sizeof(int));
		buffers = new CBufferQueue(Width, QUEUESIZE);
	}

	~CDataFileReader()
	{
		file.close();
		delete buffers;
	}

	virtual int GetWidth() { return Width; }
	virtual int GetHeight() { return Height; }
	virtual int GetChannelCount() { return channelCount; }
	virtual short* ReadLine(int line, int color)
	{
		short* buffer = buffers->Get();
		uint64_t pos = ((uint64_t)line * channelCount + color) * Width * sizeof(short) + headerSize;
		file.seekg(pos);
		file.read((char*)buffer, Width * sizeof(short));
		return buffer;
	}
};
class CMemoryFileReader : public CDataFileReader
{
	short** dataLines;
	int nCount;
public:
	CMemoryFileReader(std::string filePath) :CDataFileReader(filePath)
	{
		nCount = GetHeight() * GetChannelCount();
		dataLines = new short* [nCount];
		for (int a = 0; a < nCount; a++)
			dataLines[a] = nullptr;
	}
	
	~CMemoryFileReader()
	{
		for (int a = 0; a < nCount; a++)
			delete dataLines[a];
		delete dataLines;
	}
	virtual short* ReadLine(int line, int color) override
	{
		int y = line * GetChannelCount() + color;
		if (dataLines[y] == nullptr)
		{
			dataLines[y] = new short[GetWidth()];
			memcpy(dataLines[y],CDataFileReader::ReadLine(line, color),GetWidth() * sizeof(short));
		}
		return dataLines[y];
	}
};

class CCacheFileReader : public CDataFileReader
{
	short** dataLines;
	int nCount;
	int* queue;
	int qSize;
	int qPos;
public:
	CCacheFileReader(std::string filePath,uint32_t cacheSizeMB) :CDataFileReader(filePath)
	{
		qSize = ((uint64_t)cacheSizeMB << 20) / GetWidth();
		queue = new int[qSize];
		for (int a = 0; a < qSize; a++)
			queue[a] = -1;
		nCount = GetHeight() * GetChannelCount();
		dataLines = new short* [nCount];
		for (int a = 0; a < nCount; a++)
			dataLines[a] = nullptr;
		qPos = 0;
	}

	~CCacheFileReader()
	{
		for (int a = 0; a < nCount; a++)
			delete dataLines[a];
		delete dataLines;
		delete queue;
	}
	virtual short* ReadLine(int line, int color) override
	{
		int y = line * GetChannelCount() + color;
		if (dataLines[y] == nullptr)
		{
			dataLines[y] = new short[GetWidth()];
			memcpy(dataLines[y], CDataFileReader::ReadLine(line, color), GetWidth() * sizeof(short));
			queue[qPos++] = y;
			if (qPos == qSize) qPos = 0;
			if (queue[qPos] != -1)
			{
				delete dataLines[queue[qPos]];
				dataLines[queue[qPos]] = nullptr;
			}				
		}
		return dataLines[y];
	}
};

class Convolver : public CColorLineReader
{
	std::vector<short> kernel;
	CColorLineReader* parent;
	CBufferQueue* buffers;
public:
	Convolver(CColorLineReader* parent) :parent(parent)
	{
		buffers = new CBufferQueue(GetWidth(), QUEUESIZE);
	}
	~Convolver()
	{
		delete buffers;
	}
	void setKernel(const std::vector<short>& kernel) { this->kernel = kernel; }

	void makeGaussKernel(int radius)
	{
		std::vector<double> k;
		for (int a = -radius; a <= radius; a++)
		{
			k.push_back(My::Math::Gauss2((double)a / radius));
		}

		int size = (int) k.size();
		kernel.clear();
		for (double ky : k)
			for (double kx : k)
				kernel.push_back((short)(ky * kx * MAXVALUE));
	}

	virtual short Get(int x, int y, int c) override
	{		
		int cc = (int)sqrt(kernel.size());
		int val = parent->Get(x, y, c);
		if ((cc > 0) && (val > 0))
		{			
			for (int yy = -cc / 2; yy <= cc / 2; yy++)
			{
				for (int xx = -cc / 2; xx <= cc / 2; xx++)
				{
					if (parent->Get(x + xx, y + yy, c) == 0)
						val -= kernel[(yy + cc / 2) * cc + xx + cc / 2];
				}
			}
			if (val < 0) val = 0;
		}
		return (short)val;
	}
	virtual int GetWidth() { return parent->GetWidth(); }
	virtual int GetHeight() { return parent->GetHeight(); }
	virtual short* ReadLine(int line, int color)
	{
		short* buffer = buffers->Get();
		for (int x = 0; x < GetWidth(); x++)
			buffer[x] = Get(x, line, color);
		return buffer;
	}
};


void readertest(std::string dataFilePath, std::string outPath)
{
	CDataFileReader reader(dataFilePath);
	Convolver convolver(&reader);
	convolver.makeGaussKernel(5);
	int w = reader.GetWidth();
	int h = reader.GetHeight();
	int lastpercent = -1;
	CColorLineReader* source = static_cast<CColorLineReader*> (&convolver);	
	CDataFileWriter writer(outPath, w, h, 1);
	debug << "% xx";
	for (int y = 0; y < h; y++)
	{				
		short* buffer = source->ReadLine(y, 0);
		for (int x = 0; x < w; x++)
			buffer[x] = (4095 - buffer[x]) << 4;

		int percent = 100 * y / h;
		
		if (percent != lastpercent)
		{
			lastpercent = percent;
			debug << "\b\b" << std::setw(2) << percent+1;
		}
		writer.WriteLine(y, 0, buffer);
	}
	debug << "\nend.\n";
}

using namespace My;



class MyEngine : public My::Engine
{
public:
	MyEngine() :My::Engine()
	{}

	bool OnStart() override
	{		
		std::string inputImagePath = myfs::path("asset/printsample.tif");
		std::string dataFilePath = myfs::path("test.prn");
		std::string outPath = myfs::path("out.raw");
		//if (writetestfile(inputImagePath, dataFilePath))
		{
			StopWatch s;
			s.Start();
			readertest(dataFilePath, outPath);
			s.Stop();
			debug << "Image Process & Write Time = " << s.GetDurationS() << "seconds.\n";
		}				
		return true;
	}

	void OnDraw() override
	{

	}

};