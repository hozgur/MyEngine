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
    if (!initArrayModule())
    {
        PyMem_RawFree(program);
        program = nullptr;
        return false;
    }        
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
    //debug << "result = " << PyLong_AsLong(pResult);
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


static PyObject*
spam_system(PyObject* self, PyObject* args)
{
    const char* command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject* example_wrapper(PyObject* dummy, PyObject* args)
{

}


static PyMethodDef SpamMethods[] = {
    { "system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    "module doc", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC PyInit_spam(void)
{
    return PyModule_Create(&spammodule);
}

#include "mypyarray.h"
bool My::Py::addModule(pymodule* module)
{
    if (PyImport_AppendInittab("spam", PyInit_spam) == -1) {
        debug << "Error: could not extend in-built modules table\n";
        return false;
    }
    
    return true;
}
