#pragma once
#pragma once

#include "my.h"
#include "mypy.h"
#include "mytensor.h"

using namespace My;
class MyEngine : public My::Engine
{
public:
	MyEngine(const char* path) :My::Engine(path)
	{

	}
	std::string replace(std::string str, const std::string& from, const std::string& to) {
		size_t start_pos = str.find(from);
		if (start_pos == std::string::npos)
			return str;
		std::string newstr = str.replace(start_pos, from.length(), to);
		return newstr;
	}
	Py::dict results;
	int c = 0;
	int id = 0;
	bool OnStart() override
	{
		if (!Py::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		Py::dofile(myfs::path("user/DeepLearning/MnistTest/main.py"));
		Py::DumpGlobals();
		id = Py::dofunction("doBatch2", { (long)(400)});
		AddWindow(800, 600);
		results["id"] = 12L;
		return true;
	}

	void run()
	{
		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 20; j++)
			{
				id = Py::dofunction("doBatch2", { (long)(i * 10 + j + c * 400) });
				//int id2 = Py::dofunction("doBatch", { (long)(id) });
				//tensor<uint8_t>* tens = (tensor<uint8_t>*) GetMyObject(id2);
				tensor<uint8_t>* tens = nullptr;
				if (tens != nullptr)
				{
					for (int y = 0; y < 28; y++)
					{
						uint8_t* tData = tens->getData(1, y);
						Color* p = GetLinePointer(y + i * 28) + j * 28;
						for (int x = 0; x < 28; x++)
						{
							uint8_t t = tData[x];
							p[x] = Color(t, t, t);
						}
					}
				}

			}
		}
		c++; if (c > 100) c = 0;
	}

	void OnIdle() override
	{
		
	}

	void OnDraw() override
	{
		
	}

	void OnUpdate() override
	{
		run();
	}
};