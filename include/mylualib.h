#pragma once

typedef int (*myLuaCFunction) (void* L);
class myLualib
{
public:
virtual myLuaCFunction getLibFunction() = 0;
};

