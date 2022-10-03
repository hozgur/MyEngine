#include "my.h"
#include "mypy.h"
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <filesystem>
#include <mypytensor.h>
#include "mypyarray.h"
#include "mypyengine.h"
namespace fs = std::filesystem;

#include "mypyarray.h"

struct pyconvert
{
    PyObject* operator()(long i) const {
        return PyLong_FromLong(i);
    }
    PyObject* operator()(double f) const {
        return PyLong_FromDouble(f);
    }
    PyObject* operator()(const std::string& s) const {
        return PyUnicode_FromString(s.c_str());
    }
};
wchar_t* myPy::program = nullptr;
PyObject* gpDict;

void myPy::exit()
{
    if (program)
    {
        if (Py_FinalizeEx() < 0)        
            debug << "Error on Python Finalize.\n";
        else
            PyMem_RawFree(program);
    }
}

bool myPy::init()
{
    program = Py_DecodeLocale(myEngine::pEngine->appPath.c_str(), NULL);
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
	//gpDict Global Dictionary for python global variables.
    gpDict = PyDict_New();
    PyDict_SetItemString(gpDict, "__builtins__", PyEval_GetBuiltins());
    return true;
}

bool myPy::isInitialized()
{
    return Py_IsInitialized() != 0;
}

bool myPy::dostring(std::string content)
{        
    auto result = PyRun_String(content.c_str(), Py_file_input, gpDict, gpDict);
    if (result == nullptr)
    {
        PyErr_Print();
        return false;
    }
    return  true;
}


int myPy::call(std::string funcname, paramlist parameters)
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

    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    int result = -1;
    if (pResult != nullptr)
    {
        if (PyNumber_Check(pResult))
            result = PyLong_AsUnsignedLong(pResult);
        Py_DECREF(pResult);
    }
    else
        PyErr_Print();

    Py_DECREF(pArgs);
    return result;
}

bool myPy::checkfunction(std::string funcname)
{
    PyObject* pFunc = PyDict_GetItemString(gpDict, funcname.c_str());    
    return (pFunc != nullptr);
}

bool myPy::dofile(std::string file)
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


static PyObject* engine_test(PyObject* self, PyObject* args)
{ 
    Py_ssize_t argCount = PyTuple_Size(args);
    if (argCount > 0)
    {
        PyObject* shape = PyTuple_GetItem(args, 0);
        if (shape == NULL) { return NULL; }

        if (!PyTuple_Check(shape))
        {
            PyErr_SetString(PyExc_TypeError, "shape must be list.");
            return NULL;
        }
        Py_ssize_t shapesize = PyTuple_Size(shape);

        std::vector<int> vshape;
        for (int i = 0; i < shapesize; i++) {
            PyObject* dim = PyTuple_GetItem(shape, i);
            /* Check if temp_p is numeric */
            if (!PyNumber_Check(dim)) {
                PyErr_SetString(PyExc_TypeError, "Non-numeric argument.");
                return NULL;
            }
            PyObject* longNumber = PyNumber_Long(dim);
            vshape.push_back((int)PyLong_AsUnsignedLong(longNumber));
            
            debug << vshape.back();
            Py_DECREF(longNumber);            
            if (PyErr_Occurred()) { return NULL; }
        }
        Py_DECREF(shape);

        if (argCount > 1)
        {
            PyObject* typeString = PyTuple_GetItem(args, 1);
            Py_ssize_t size;
            const char* ptr = PyUnicode_AsUTF8AndSize(typeString, &size);
            debug << ptr;
            Py_DECREF(typeString);
        }

        if (argCount > 2)
        {
            PyObject* _itemSize = PyTuple_GetItem(args, 2);
            long itemSize = PyLong_AsLong(_itemSize);
            debug << itemSize;
            Py_DECREF(_itemSize);
        }
    }
    PyObject* longNumber = Py_BuildValue("i", 0);
    return longNumber;
}

template<>
void myPy::setglobal(const char* name, const int& val)
{
    PyDict_SetItemString(gpDict, name, PyLong_FromLong(val));
}