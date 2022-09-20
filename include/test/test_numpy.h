#pragma once
#pragma once
#include "mypy.h"
using namespace My;

void test5()
{
    myPy::init();
    myPy::dict locals;
    locals["a"] = 45L;
    myPy::dofile(myfs::path("user/py_test.py"));
   // Py::dofunction("test_numpy", {});
}

class MyEngine : public My::myEngine
{
public:

    MyEngine(const char* path) :My::myEngine(path)
    {        
    }

    bool OnStart() override
    {
        test5();
        return true;
    }
};