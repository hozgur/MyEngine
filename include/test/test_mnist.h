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
		if (!Py::init())
		{
			debug << "Py Error!\n";
			exit(1);
		}
	}

	bool OnStart() override
	{
		Py::dofile(myfs::path("user/DeepLearning/MnistTest/main.py"));
		Py::dict results;
		results["id"] = 12L;
		Py::dostring("id = tensor.fromBuffer(dataset1[0][0].tobytes(),(1,28,28))", {}, results);
		
		mytensor<uint8_t> *tensor = (mytensor<uint8_t>*) GetMyObject(std::get<long>(results["id"]));
		debug << *tensor;
		return true;
	}
};