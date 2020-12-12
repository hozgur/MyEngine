#pragma once
#include "my.h"
#include "windows\windowscore.h"
#include "windows\windowsplatform.h"
#include "ilay_io.h"
#include "ilay_convolver.h"


bool writetestfile(std::string inputFile, std::string outputFile, int scaleshift = 0)
{
	My::StopWatch s;
	s.Start();
	My::Surface* surface = My::LoadSurface(myfs::s2w(inputFile));
	s.Stop();
	debug << "Image File Read Time = " << s.GetDurationS() << " seconds.\n";
	if (surface == nullptr)
	{
		debug << "error on load!";
		return false;
	}

	int w = surface->GetWidth() << scaleshift;
	int h = surface->GetHeight() << scaleshift;
	int d = surface->GetDepth();
	debug << "Width = " << w << " Height = " << h << " Depth = " << d << "\n";
	if (d != 8)
	{
		debug << "Image must be 1 (8bit) channel.\n";
		return false;
	}
	ilay::FileWriter writer(outputFile, w, h, 1);
	std::vector<short> buffer(w);
	s.Start();
	for (int y = 0; y < h; y++)
	{
		uint8_t* p = (uint8_t*)surface->GetLinePointer(y >> scaleshift);
		for (int c = 0; c < 1; c++)
		{
			for (int x = 0; x < w; x++)
				buffer[x] = MAXVALUE - p[x >> scaleshift] * MAXVALUE / 255;
			writer.WriteLine(y, c, buffer.data());
		}
	}
	s.Stop();
	debug << "Data File Write Time = " << s.GetDurationS() << " seconds.\n";
	return true;
}
using namespace ilay;

void readertest(std::string dataFilePath, std::string outPath)
{
	int radius = 5;	
	int bufferCount = 2 * radius + 1;
	#pragma omp parallel
	{
		FileReader reader(dataFilePath, bufferCount);
		VConvolver vconvolver(&reader, radius, bufferCount);
		HConvolver hconvolver(&vconvolver, radius, 1);
		int w = reader.GetWidth();
		int h = reader.GetHeight();
		FileWriter writer(outPath, w, h, 1);
		#pragma omp for
		for (int y = 0; y < h; y++)
		{
			short* buffer = hconvolver.ReadLine(y, 0);

			for (int x = 0; x < w; x++)
				buffer[x] = (MAXVALUE - buffer[x]) << 4;
			
			writer.WriteLine(y, 0, buffer);
		}
	}
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
		//if (writetestfile(inputImagePath, dataFilePath,0))
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