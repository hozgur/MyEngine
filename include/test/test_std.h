#pragma once
#pragma once
#include "mypy.h"
using namespace My;

class test
{
public:
    void testfunction(int val)
    {
        debug << val;
    }

};


class MyEngine : public My::myEngine
{
public:
    
    MyEngine(const char* path) :My::myEngine(path)
    {
        myCommand q;
        if (commandQueue.pop(q))
            debug << q.id;
        commandQueue.push({ 12,myCommands::Navigate,{"tata"} });
        if (commandQueue.pop(q))
            debug << q.id;
        
    }        
    
    bool OnStart() override
    {
     
        return true;
    }
    
};