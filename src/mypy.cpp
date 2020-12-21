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
#include <py_mytensor.h>
#include "mypyarray.h"
namespace fs = std::filesystem;
using namespace My;

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
    addModule(nullptr);
    Py_Initialize();    
    return true;
}

bool My::Py::isInitialized()
{
    return Py_IsInitialized() != 0;
}

bool My::Py::dostring(std::string content)
{        
    return PyRun_SimpleString(content.c_str()) == 0;
}

bool My::Py::dostring(std::string content, dict locals)
{
    PyObject* pDict = PyDict_New();
    if (!pDict) return NULL;
    PyDict_SetItemString(pDict, "__builtins__", PyEval_GetBuiltins());
    pyconvert pc;
    for (dictItem element : locals)
        PyDict_SetItemString(pDict, element.first.c_str(), std::visit(pc, element.second));
    return PyRun_String(content.c_str(), Py_file_input, pDict, pDict) != NULL;
}
bool My::Py::dostring(std::string content, dict locals, dict &result)
{    
    PyObject* pDict = PyDict_New();    
    if (!pDict) return NULL;

    PyDict_SetItemString(pDict, "__builtins__", PyEval_GetBuiltins());
    pyconvert pc;
    for (dictItem element : locals)
        PyDict_SetItemString(pDict, element.first.c_str(), std::visit(pc, element.second));

    if (PyRun_String(content.c_str(), Py_file_input, pDict, pDict) != NULL)
    {
        for (dictItem element : result)        
        {
            const char* key = element.first.c_str();
            auto value = element.second;
            
            if (std::get_if<double>(&value))
            {
                result[key] = PyFloat_AsDouble(PyDict_GetItemString(pDict, key));
            }else
            if (std::get_if<long>(&value))
            {
                result[key] = PyLong_AsLong(PyDict_GetItemString(pDict, key));
            }else
            if (std::get_if<std::string>(&value))
            {
                result[key] = PyUnicode_AsUTF8(PyDict_GetItemString(pDict, key));
            }
        }
    }
    return true;
}

bool My::Py::dofunction(std::string funcname, paramlist parameters)
{
    PyObject *mainModule = PyImport_ImportModule("__main__");
    PyObject* pFunc = PyObject_GetAttrString(mainModule, funcname.c_str());
    PyObject* pArgs = PyTuple_New(parameters.size());
    int i = 0;
    pyconvert pc;
    for (std::variant v : parameters)    
        PyTuple_SetItem(pArgs, i++, std::visit(pc, v));
        
    PyObject *pResult = PyObject_CallObject(pFunc, pArgs);    
    if (PyObject_CheckBuffer(pResult))
    {
        Py_buffer buffer;
        if (PyObject_GetBuffer(pResult, &buffer, NULL) == 0)
        {
            debug << buffer.len << "\n";
            debug << buffer.strides << "\n";
            PyBuffer_Release(&buffer);
        }
    }
    Py_DECREF(pResult);
    Py_DECREF(pFunc);
    Py_DECREF(pArgs);
    return true;
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
    
    return PyRun_SimpleString(content.c_str()) == 0;
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

static PyMethodDef EngineMethods[] = {
    { "engine_test",  engine_test, METH_VARARGS,
     "Test method for engine module."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef engineModule = {
    PyModuleDef_HEAD_INIT,
    "MyEngine",   /* name of module */
    "module doc", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    EngineMethods
};

PyMODINIT_FUNC PyInit_MyEngine(void)
{
    return PyModule_Create(&engineModule);
}

bool My::Py::addModule(pymodule* module)
{
    if (PyImport_AppendInittab("MyEngine", PyInit_MyEngine) == -1) {
        debug << "Error: could not extend in-built modules table\n";
        return false;
    }
    return true;
}
