#pragma once
#include "mypy.h"

void test1()
{
    myStopWatch s;
    s.Start();
    for (int a = 0; a < 10000; a++)
    {
        myPy::dict locals;
        locals["a"] = 45L;
        myPy::dict result;
        result["outa"] = 0L;
        myPy::dostring("outa = 25 * a\n", locals, result);
        //debug << std::get<long>(result["outa"]);
    }
    s.Stop();
    debug << s.GetDurationStr();
}

void test2()
{
    myPy::dofile(myfs::path("user/py_test.py"));
    myPy::call("test_function", { 33. });
}

struct SampleVisitor
{
    void operator()(int i) const {
        std::cout << "int: " << i << "\n";
    }
    void operator()(float f) const {
        std::cout << "float: " << f << "\n";
    }
    void operator()(const std::string& s) const {
        std::cout << "string: " << s << "\n";
    }
};


void test3()
{
    std::variant<int, float, std::string> intFloatString;
    static_assert(std::variant_size_v<decltype(intFloatString)> == 3);

    // default initialized to the first alternative, should be 0
    std::visit(SampleVisitor{}, intFloatString);

    SampleVisitor s;
    std::visit(s,intFloatString);

    // index will show the currently used 'type'
    std::cout << "index = " << intFloatString.index() << std::endl;
    intFloatString = 100.0f;
    std::cout << "index = " << intFloatString.index() << std::endl;
    intFloatString = "hello super world";
    std::cout << "index = " << intFloatString.index() << std::endl;
}

void test4()
{
    myPy::dict locals;
    locals["a"] = 45L;    
    myPy::dostring("outa = 25 * a\n", locals);
}

class MyEngine : public myEngine
{
public:

    MyEngine(const char* path) :myEngine(path) 
    {
        //py.addModule(nullptr);
        myPy::init();
    }

    bool OnStart() override
    {
        test4();
        return true;
    }

};