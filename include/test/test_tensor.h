#pragma once
#include "my.h"
#include "mytensor.h"
using namespace My;
class MyEngine : public My::Engine
{
public:

	MyEngine(const char* path) :My::Engine(path)
	{
		//mytensorImpl<int> tensor({ 3, 5, 5 });
		mytensorImpl<int> tensor({});
		debug << tensor;
		*tensor.pData = 123;
		debug << "Strides = ";
		for (int64_t s : tensor.strides())
			debug << s << " ";

		tensor = mytensorImpl<int>({});

		debug << tensor;

		debug << "Strides = ";
		for (int64_t s : tensor.strides())
			debug << s << " ";
	}

	bool OnStart() override
	{
		return true;
	}
};