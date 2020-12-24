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
	bool OnStart() override
	{
		if (!Py::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
		Py::dofile(myfs::path("user/DeepLearning/MnistTest/main.py"));
		AddWindow(800, 600);		
		results["id"] = 12L;	
		return true;
	}

	void OnIdle() override
	{
		//Py::dostring("runBatch()");
		//std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}

	void OnDraw() override
	{
		
		for (int i = 0; i < 20; i++)
		{
			for (int j = 0; j < 20; j++)
			{
				std::string line = "id = tensor.fromBuffer(dataset1[$n][0].tobytes(),(1,28,28),'byte')";
				std::string line2 = replace(line, "$n", std::to_string(i * 10 + j + c * 400));
				Py::dostring(line2, {}, results);
				tensor<uint8_t>* tens = (tensor<uint8_t>*) GetMyObject(std::get<long>(results["id"]));
				uint8_t* tDat = tens->getData(0, 0);
				for (int y = 0; y < 28; y++)
				{
					//uint8_t* tData = tDat + 28 * y;
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
		c++; if (c > 100) c = 0;
	}
};