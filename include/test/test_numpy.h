#pragma once
#pragma once
#include "mypy.h"
using namespace My;

void test5()
{
    Py::addModule(nullptr);
    Py::init();
    Py::dict locals;
    locals["a"] = 45L;
    Py::dofile(myfs::path("user/py_test.py"));
   // Py::dofunction("test_numpy", {});
}

class MyEngine : public My::Engine
{
public:

    MyEngine(const char* path) :My::Engine(path)
    {        
    }

    bool OnStart() override
    {
        test5();
        return true;
    }
};