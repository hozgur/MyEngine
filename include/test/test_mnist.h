#pragma once

#include "my.h"
#include "mypy.h"
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
		return true;
	}
};