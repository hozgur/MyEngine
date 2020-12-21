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
		debug << "\n";
		const int itemsize = 33;
		mytensorImpl<char[itemsize]> tensor2 = mytensorImpl<char[itemsize]>({3,24,50});

		debug << tensor2;
		strcpy_s(tensor2.getData(0, 0)[0],itemsize, "Hello!");
		debug << *tensor2.getData(0,0);
		debug << "Strides = ";
		for (int64_t s : tensor2.strides())
			debug << s << " ";
	}

	bool OnStart() override
	{
		return true;
	}
};