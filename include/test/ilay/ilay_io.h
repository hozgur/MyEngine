#pragma once
#include <omp.h>
#include <mutex>
namespace ilay
{
	std::mutex mtx;
	//static std::vector<std::string> objects;

	void addString(std::string str)
	{
		std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
		lck.lock();
		int id = omp_get_thread_num();
		std::cout << "thread #" << id << " class = " << str << '\n';
		lck.unlock();
	}
	class LineReader
	{
	public:
		virtual int GetWidth() = 0;
		virtual int GetHeight() = 0;
		virtual short* ReadLine(int line, int color) = 0;		
	};

	class LineWriter
	{
	public:
		virtual int GetWidth() = 0;
		virtual int GetHeight() = 0;
		virtual void WriteLine(int line, int color, const short* buffer) = 0;
	};

	class LineBuffer
	{
		short** buffers;
		int bufferWidth;
		int bufferCount;
		std::atomic<int> currentBuffer;
	public:
		LineBuffer(int bufferwidth, int buffercount) : bufferWidth(bufferwidth), bufferCount(buffercount)
		{
			buffers = new short* [buffercount];
			for (int a = 0; a < buffercount; a++)
				buffers[a] = new short[bufferwidth];
			currentBuffer = 0;
		}

		~LineBuffer()
		{
			for (int a = 0; a < bufferCount; a++)
				delete buffers[a];
			delete buffers;
		}

		short* Get()
		{			
			return buffers[currentBuffer++ % bufferCount];
		}

		short* Get(int index)
		{
			return buffers[mod(index,bufferCount)];
		}
	};

	class FileReader : public LineReader
	{
	protected:
		int Width;
		int Height;
		int channelCount;
		std::ifstream file;
		const int headerSize = 3 * sizeof(int);
		LineBuffer* buffers;
		
	public:
		FileReader(std::string filePath,int bufferCount = 1)
		{
			addString("File Reader");
			file.open(filePath, std::ios::in | std::ios::binary);
			file.read((char*)&Width, sizeof(int));
			file.read((char*)&Height, sizeof(int));
			file.read((char*)&channelCount, sizeof(int));
			buffers = new LineBuffer(Width, bufferCount);
		}

		virtual ~FileReader()
		{
			file.close();
			delete buffers;
		}

		virtual int GetWidth() { return Width; }
		virtual int GetHeight() { return Height; }
		virtual int GetChannelCount() { return channelCount; }
		virtual short* ReadLine(int line, int color)
		{
			//addString("ReadLine");
			short* buffer = buffers->Get();
			uint64_t pos = ((uint64_t)mod(line, GetHeight()) * channelCount + color) * Width * sizeof(short) + headerSize;
			file.seekg(pos);
			file.read((char*)buffer,Width * sizeof(short));
			return buffer;
		}
	};

	class FileWriter : public LineWriter
	{
		int Width;
		int Height;
		int channelCount;
		std::ofstream file;
		const int headerSize = 3 * sizeof(int);
	public:
		FileWriter(std::string filePath, int width, int height, int channelcount) :Width(width), Height(height), channelCount(channelcount)
		{
			addString("File Writer");
			file.open(filePath, std::ios::out | std::ios::binary);
			file.write((char*)&width, sizeof(int));
			file.write((char*)&height, sizeof(int));
			file.write((char*)&channelcount, sizeof(int));
		}

		~FileWriter()
		{
			file.close();
		}

		void Close() { file.close(); }

		virtual int GetWidth() { return Width; }
		virtual int GetHeight() { return Height; }
		virtual void WriteLine(int line, int color, const short* buffer)
		{
			uint64_t pos = ((uint64_t)line * channelCount + color) * Width * sizeof(short) + headerSize;
			file.seekp(pos);
			file.write((char*)buffer, Width * sizeof(short));
		}
	};
}
