#pragma once
#include "my.h"
#include "molecular/molecular2.h"

using namespace My;

class test_perf
{
	StopWatch s;
	std::string name;
public:
	void (*test_func)(void);
	void (*reference_func)(void);

	test_perf(std::string name):test_func(nullptr),reference_func(nullptr),name(name){}

	void run()
	{
		debug << name << " test starting.\n";
		test_func();
	}

};



class MyEngine : public My::Engine
{
public:
	typedef int testtype;
	const int width = 800;
	const int height = 600;
	std::vector<pixel<testtype>> pixels;
	MyEngine(const char* path) :My::Engine(path) {}

	float frand() { return (float)std::rand() / RAND_MAX; }
	void InitDots()
	{
		int size = (width / 2) * (height / 2);
		debug << "Dot Count = " << size << "\n";
		pixels.reserve(size);
		for (int y = 0; y < height; y+=2)
		{
			for (int x = 0; x < width; x+=2)
			{
				pixels.push_back(pixel<testtype>(x, y, Color::Random(),(testtype) frand(), (testtype) frand()));
			}
		}
	}
	bool OnStart() override
	{
		//AddWindow(width, height, 1, 1);

		test_perf t("integer mod test");
		t.test_func = []()
		{
			debug << "test";
		};
		t.run();
		InitDots();
		//for (int a = 0; a < 10; a++)
		//	test();
		return true;
	}
	void MoveDot(int nDot)
	{

	}

	void test()
	{
		StopWatch s;
		s.Start();
		const int testCount = 10000;
		double t1, t2;
		long long sum1 = 0;
		for (int a = 0; a < testCount; a++)
		{
			sum1 += draw();
		}
		s.Stop();
		t1 = s.GetDurationMS();
		long long sum2 = 0;
		s.Start();
		for (int a = 0; a < testCount; a++)
		{
			sum2 += draw_reference();
		}
		s.Stop();
		t2 = s.GetDurationMS();
		debug << "Duration = " << t1 << "ms. Reference = " << t2 << "ms. Diff = " << t1 - t2 << "ms. %" << 100 * (t1 - t2) / t2 << "\n";
		debug << "Sum  = " << sum1 << " Sum Ref = " << sum2 << "\n";
	}

	int draw()
	{
		int sum = 0;
		for (pixel<testtype>& p : pixels)
		{
			int x = ((int)p.x);//% width;
			int y = ((int)p.y);//% height;
			sum += x + y;
		}
		return sum;
	}

	int draw_reference()
	{	
		int sum = 0;
		for (pixel<testtype>& p : pixels)
		{
			int x = (int)p.x;
			int y = (int)p.y;
			sum += x + y;
		}
		return sum;
	}

	

	void OnDraw() override
	{
		
	}

	void OnUpdate() override
	{

	}

	void OnExit() override
	{

	}
};