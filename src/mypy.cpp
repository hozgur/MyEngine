#include "my.h"
#include "mypy.h"
#define PY_SSIZE_T_CLEAN
#ifdef _DEBUG
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include <filesystem>
#include "py_core.h"
#include "py_mytensor.h"
#include "py_engine.h"

namespace fs = std::filesystem;
wchar_t* My::Py::program = nullptr;

void My::Py::exit()
{
    if (program)
    {
        if (Py_FinalizeEx() < 0)        
            debug << "Error on Python Finalize.\n";
        else
            PyMem_RawFree(program);
    }
}
PyObject* gpDict;
bool My::Py::init()
{
    program = Py_DecodeLocale(My::Engine::pEngine->appPath.c_str(), NULL);
    if (program == nullptr)        
        return false;

    Py_SetProgramName(program);
    if (!initTensorModule())
    {
        PyMem_RawFree(program);
        program = nullptr;
        return false;
    }    
    if (!addEngineModule())
        debug << "Error on adding Python MyEngine module.\n";
    Py_Initialize();
    gpDict = PyDict_New();
    PyDict_SetItemString(gpDict, "__builtins__", PyEval_GetBuiltins());
    return true;
}

bool My::Py::isInitialized()
{
    return Py_IsInitialized() != 0;
}

bool My::Py::dostring(std::string content)
{            
    auto result = PyRun_String(content.c_str(), Py_file_input, gpDict, gpDict);
    if (result == nullptr)
    {
        PyErr_Print();
        return false;
    }
    return  true;
}

void My::Py::DumpGlobals()
{
    PyObject* keys = PyDict_Keys(gpDict);
    Py_ssize_t size = PyList_Size(keys);
    for (int a = 0; a < size; a++)
    {
        PyObject* item = PyList_GetItem(keys, a);
        PyObject* strItem = PyObject_Str(item);
        debug << PyUnicode_AsUTF8(strItem) << "\n";
    }
}
int My::Py::dofunction(std::string funcname, paramlist parameters)
{
    PyObject* pFunc = PyDict_GetItemString(gpDict, funcname.c_str());
    if (pFunc == nullptr)
    {
        debug << funcname << " not found.";
        return -1;
    }
    PyObject* pArgs = PyTuple_New(parameters.size());
    int i = 0;
    static pyconvert pc;
    for (std::variant v : parameters)    
        PyTuple_SetItem(pArgs, i++, std::visit(pc, v));
        
    PyObject *pResult = PyObject_CallObject(pFunc, pArgs);
    int result = -1;
    if (pResult != nullptr)
    {
        if(PyNumber_Check(pResult))
            result = PyLong_AsUnsignedLong(pResult);
        Py_DECREF(pResult);
    }
    else
        PyErr_Print();     
    
    Py_DECREF(pArgs);
    return result;
}

bool My::Py::checkfunction(std::string funcname)
{
    PyObject* mainModule = PyImport_ImportModule("__main__");
    PyObject* pFunc = PyObject_GetAttrString(mainModule, funcname.c_str());
    if (pFunc == NULL)        
        return false;
    Py_DECREF(pFunc);
    return true;
}

bool My::Py::dofile(std::string file)
{
    std::ifstream ifs(file);
    std::string content((std::istreambuf_iterator<char>(ifs)),
        (std::istreambuf_iterator<char>()));
    
    auto result = PyRun_String(content.c_str(), Py_file_input, gpDict, gpDict);
    if (result == nullptr)
    {
        PyErr_Print();
        return false;
    }
    return  true;
}

template<>
int My::Py::getglobal(const char* name)
{
    return PyLong_AsLong(PyDict_GetItemString(gpDict, name));
}

template<>
void My::Py::setglobal(const char* name, const int& val)
{
    PyDict_SetItemString(gpDict, name, PyLong_FromLong(val));
}