// my.cpp
#include "python/python-header.h"

int main(int argc, char *argv[])
{   
	pyEngine engine((const char *)argv[0],argc,(const char**) argv);
	engine.Start();
    return 0;
}
