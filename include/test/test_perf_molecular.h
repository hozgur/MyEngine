#pragma once
#include "my.h"
#include "molecular/molecular2.h"

using namespace My;

#define CLR_RED "\33[91m"
#define CLR_GRN "\33[92m"
#define CLR_BLU "\33[94m"
#define CLR_DEF "\33[0m"

class test_perf
{
	
	std::string name;
	double duration = 5.;
	
public:
	double (*test_func)(int64_t);
	double (*reference_func)(int64_t);

	void SetDuration(int seconds) { duration = (double)seconds; }
	test_perf(std::string name):test_func(nullptr),reference_func(nullptr),name(name){}

	void run()
	{
		const int width = 70;
		const int loopCount = width;
		debug << name << "\n";
		debug << std::string(width,'*') << "\n";
		debug << CLR_GRN "Test starting.\n" CLR_DEF;
		debug << std::string(width, '*') << "\n";
		StopWatch s;
		s.Start();
		int64_t mul = 1000;
		for(int64_t a = 0;a<mul;a++)
			test_func(a);
		double t = s.Stop();
		
		if (t < 1e-6 )
		{
			mul = (int64_t) 1e9;
			s.Start();
			for(int64_t a = 0; a<mul;a++)
				test_func(a);
			t = s.Stop();
		}
		if (t == 0)
		{
			debug << "Error on system!! Please check timer and test function.\n";
			return;
		}

		int64_t count = (int64_t)(mul * duration / t /loopCount);

		debug << "Duration = " << t/mul << " Test Count = " << count << " mul = " << mul << "\n";
		t = 0;
		double tr = 0;

		double temp = 0;
		for (int a = 0; a < loopCount; a++)
		{
			s.Start();
			for(int64_t i=0; i < count; i++)
				temp += test_func(i);
			t += s.Stop();
			s.Start();
			for (int64_t i = 0; i < count; i++)
				temp += reference_func(i);
			tr += s.Stop();
			debug << "*";
		}
		debug << "\n";
		// temporary save for preventing optimizer.
		std::ofstream tmpfile;
		tmpfile.open("temp.txt");
		tmpfile << temp << "\n";
		tmpfile.close();

		double tdiff = (t - tr)/tr;
		debug << "Reference Duration = " << tr << "s. Test Duration = " << t << "s.\n";
		debug << std::string(width, '*') << "\n";
		if (tdiff < 0) debug << CLR_RED; else debug << CLR_GRN;
		debug << "Difference = %" << 100 * tdiff  << CLR_DEF "\n";
		debug << std::string(width, '*') << "\n";
	}

};


int table[65536];
double max = pow(65535, 2.9);

class MyEngine : public My::Engine
{
public:
	typedef int testtype;
	const int width = 800;
	const int height = 600;
	
	MyEngine(const char* path) :My::Engine(path) {}

	float frand() { return (float)std::rand() / RAND_MAX; }

	
	void InitDots()
	{
	
	}
	bool OnStart() override
	{
		//AddWindow(width, height, 1, 1);
		{
			test_perf t("Integer Mod Test");
			t.test_func = [](int64_t a) -> double
			{
				return (double)(a & 1145);
			};

			t.reference_func = [](int64_t a) -> double
			{				
				return (double)(a % 1145);
			};
			t.run();
		}

		{
			test_perf t("Double - Integer Mod Test");
			t.test_func = [](int64_t a) -> double
			{
				double f = (double)a;
				return fmod(f, 1145);
			};

			t.reference_func = [](int64_t a) -> double
			{
				double f = (double)a;
				int64_t a1 = (int64_t)f;
				return (double)(a1 & 1145);
			};
			t.run();
		}
		
		{
			test_perf t("Float - Double Mod Test");
			t.test_func = [](int64_t a) -> double
			{
				double f = (double)a;
				return fmod(f, 1145);
			};

			t.reference_func = [](int64_t a) -> double
			{
				float f = (float)a;
				return fmod(f, 1145.f);
			};
			t.run();
		}

		{
			test_perf t("Exp - Table Test");			
			for (int a = 0; a < 65536; a++)
				table[a] = (int)(INT_MAX * pow(a, 2.9)/max);
			t.test_func = [](int64_t a) -> double
			{
				double f = (double)a;
				return pow(f, 2.9);
			};

			t.reference_func = [](int64_t a) -> double
			{
				return (double)(table[a & 65535] / INT_MAX * max);
			};
			t.run();
		}
		return true;
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