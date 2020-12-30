#pragma once

namespace My
{
    namespace Py
    {
        static PyObject* engine_path(PyObject* self, PyObject* args)
        {
            const char* path = nullptr;
            if (PyArg_ParseTuple(args, "s", &path) == 0)
            {
                PyErr_SetString(PyExc_TypeError, "Invalid path argument string.");
                Py_RETURN_NONE;
            }
            return PyUnicode_FromString(myfs::path(path).c_str());
        }

        static PyObject* engine_import(PyObject* self, PyObject* args)
        {
            const char* file = nullptr;
            if (PyArg_ParseTuple(args, "s", &file) == 0)
            {
                PyErr_SetString(PyExc_TypeError, "Invalid path argument string.");
                Py_RETURN_NONE;
            }
            if(dofile(myfs::path(file).c_str()))
                Py_RETURN_NONE;
            else
            {
                std::string err = "Error on importing file ";
                err = err + file;
                PyErr_SetString(PyExc_TypeError, err.c_str());
                Py_RETURN_NONE;
            }
        }

        static PyObject* engine_getbackground(PyObject* self, PyObject* args)
        {
            Color* pColor = Engine::pEngine->background->readLine(0);
            int width = Engine::pEngine->background->getWidth();
            int stride = Engine::pEngine->background->getWidth() * sizeof(Color);
            int height = Engine::pEngine->background->getHeight();
            int size = stride * height;


            pytensor* pytens = (pytensor*)pytensorType.tp_alloc(&pytensorType, 0);
            if (pytens)
            {
                std::pair<char, int> p = { 'b' , 1 };                
                void* tensor = pair2tensor(p, {height,width,4}, pColor, size);
                pytens->tensorId = Engine::pEngine->SetMyObject((object*)tensor);
                pytens->type[0] = p.first;
                pytens->type[1] = 0;
                pytens->buffer = 0;

                return (PyObject*)pytens;
            }            
            else
            {
                std::string err = "Error on creating tensor.";                
                PyErr_SetString(PyExc_TypeError, err.c_str());
                Py_RETURN_NONE;
            }
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
             { "Path",  engine_path, METH_VARARGS,
             "Combine MyEngine Root Path with your relative path inside of Root."},
             { "Import",  engine_import, METH_VARARGS,
             "Import file."},
             { "GetBackground",  engine_getbackground, METH_NOARGS,
             "Get Background Image Tensor."},
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

        bool addEngineModule()
        {
            if (PyImport_AppendInittab("MyEngine", PyInit_MyEngine) == -1) {
                debug << "Error: could not extend in-built modules table\n";
                return false;
            }
            return true;
        }

    }
}