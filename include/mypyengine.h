#pragma once

namespace myPy
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
        if ((PyArg_ParseTuple(args, "s", &file) == 0) && (file != nullptr))
        {
            PyErr_SetString(PyExc_TypeError, "Invalid path argument string.");
            Py_RETURN_NONE;
        }


        if(dofile(myfs::path(file)))
            Py_RETURN_NONE;
        else
        {
            std::string err = "Error on importing file ";
            err = err + file;
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
    }

    static PyObject* engine_sendMessage(PyObject* self, PyObject* args)
    {
        const char* id = nullptr;
        const char* msg = nullptr;
        if (PyArg_ParseTuple(args, "ss", &id,&msg) == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments.");
            Py_RETURN_NONE;
        }
        myString json = std::format("{{\"id\":\"{}\",\"msg\":\"{}\"}}", id,msg);
        myEngine::pEngine->OnMessageReceived(invalidHandle, json);
        Py_RETURN_NONE;
    }

    static PyObject* engine_sendWebMessage(PyObject* self, PyObject* args)
    {
		int webId = 0;
        const char* id = nullptr;
        const char* msg = nullptr;
        if (PyArg_ParseTuple(args, "iss", &webId, &id, &msg) == 0)
        {
            PyErr_SetString(PyExc_TypeError, "Invalid arguments.");
            Py_RETURN_NONE;
        }
        myString json = std::format("{{\"id\":\"{}\",\"msg\":\"{}\"}}", id, msg);
        myEngine::pEngine->PostWebMessage(webId, json);
        Py_RETURN_NONE;
    }

    static PyObject* engine_getbackground(PyObject* self, PyObject* args)
    {
        if (myEngine::pEngine->background == nullptr)
        {
            std::string err = "No valid background.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        myColor* pColor = myEngine::pEngine->background->readLine(0);
        int width = myEngine::pEngine->background->getWidth();
        int stride = myEngine::pEngine->background->getWidth() * sizeof(myColor);
        int height = myEngine::pEngine->background->getHeight();
        int size = stride * height;

        pytensor* pytens = (pytensor*)PyObject_CallObject((PyObject*)&pytensorType, args);
        if (pytens)
        {
            myEngine::pEngine->removeObject(pytens->tensorId);
            pytens->tensorId = invalidHandle;
            if (pytens->buffer) PyBuffer_Release(pytens->buffer);
            std::pair<char, int> p = { 'b' , 1 };                
            void* tensor = pair2tensor(p, {height,width,4}, pColor, size);
            pytens->tensorId = myEngine::pEngine->setObject((myObject*)tensor);
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
    // AddMainWindow function
	// Arguments
	// 1. window title
	// 2. Width
	// 3. Height
	// 4. pixelWidth
	// 5. pixelHeight
	// 6. fullScreen
	// returns true or false
    static PyObject* engine_addmainwindow(PyObject* self, PyObject* args) {
        Py_ssize_t argCount = PyTuple_Size(args);
        if (argCount < 3) {                
            std::string err = "AddMainWindow needs parameters.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
		const char* title = nullptr;
		title = PyUnicode_AsUTF8(PyTuple_GetItem(args, 0));
        long width  = PyLong_AsLong(PyTuple_GetItem(args, 1));
        long height = PyLong_AsLong(PyTuple_GetItem(args, 2));
        long pixelWidth = 1;
		long pixelHeight = 1;
        bool fullScreen = false;
		if ((width <= 0) || (width > MAX_WIDTH)) {
			std::string err = "Invalid width.";
			PyErr_SetString(PyExc_TypeError, err.c_str());
			Py_RETURN_NONE;
		}
		if ((height <= 0) || (height > MAX_HEIGHT)) {
			std::string err = "Invalid height.";
			PyErr_SetString(PyExc_TypeError, err.c_str());
			Py_RETURN_NONE;
		}
		if (argCount > 4) {
			long pixelWidth = PyLong_AsLong(PyTuple_GetItem(args, 3));
			long pixelHeight = PyLong_AsLong(PyTuple_GetItem(args, 4));
			if ((pixelWidth <= 0) || (pixelWidth > MAX_PIXELWIDTH) || (pixelWidth * width > MAX_WIDTH)) {
				std::string err = "Invalid pixelWidth.";
				PyErr_SetString(PyExc_TypeError, err.c_str());
				Py_RETURN_NONE;
			}
			if ((pixelHeight <= 0) || (pixelHeight > MAX_PIXELHEIGHT) || (pixelHeight * height > MAX_HEIGHT)) {
				std::string err = "Invalid pixelHeight.";
				PyErr_SetString(PyExc_TypeError, err.c_str());
				Py_RETURN_NONE;
			}			
		}
        if (argCount > 5) {			
			fullScreen = PyObject_IsTrue(PyTuple_GetItem(args, 5));
		}
		
		bool ok = myEngine::pEngine->AddMainWindow(width, height, pixelWidth, pixelHeight, fullScreen);
		if(ok)
		    myEngine::pEngine->SetWindowTitle(title);
		
		return PyBool_FromLong(ok);
    }
	//AddWebView
	//Arguments:
	//1. x
	//2. y
	//3. width
	//4. height
	//5. anchor
	//6. url (optional
    static PyObject* engine_addwebview(PyObject* self, PyObject* args) {
        Py_ssize_t argCount = PyTuple_Size(args);
        if (argCount < 4) {
            std::string err = "AddWebView needs parameters.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        long x = PyLong_AsLong(PyTuple_GetItem(args, 0));
        long y = PyLong_AsLong(PyTuple_GetItem(args, 1));
        long width = PyLong_AsLong(PyTuple_GetItem(args, 2));
        long height = PyLong_AsLong(PyTuple_GetItem(args, 3));
        long anchor = 0;
        if (argCount > 4) {
            anchor = PyLong_AsLong(PyTuple_GetItem(args, 4));
        }
        if ((anchor < 0) || (anchor > 15)) {
            anchor = 0;
        }
        if ((x < 0) || (x > MAX_WIDTH)) {
            std::string err = "Invalid x.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        if ((y < 0) || (y > MAX_HEIGHT)) {
            std::string err = "Invalid y.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        if ((width <= 0) || (width > MAX_WIDTH)) {
            std::string err = "Invalid width.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        if ((height <= 0) || (height > MAX_HEIGHT)) {
            std::string err = "Invalid height.";
            PyErr_SetString(PyExc_TypeError, err.c_str());
            Py_RETURN_NONE;
        }
        const char* url = nullptr;
        if (argCount > 5) {            
            url = PyUnicode_AsUTF8(PyTuple_GetItem(args, 5));
        }
        myHandle viewId = myEngine::pEngine->AddWebView(x, y, width, height, anchor);
		if(url)
			myEngine::pEngine->Navigate(viewId, url);
		
		return PyLong_FromLong(viewId);
	}
	//Remove WebView
	//Arguments:
	//1. viewId
	static PyObject* engine_removewebview(PyObject* self, PyObject* args) {
		Py_ssize_t argCount = PyTuple_Size(args);
		if (argCount < 1) {
			std::string err = "Error unknown id. Please specify viewId.";
			PyErr_SetString(PyExc_TypeError, err.c_str());
			Py_RETURN_NONE;
		}
		long viewId = PyLong_AsLong(PyTuple_GetItem(args, 0));		
        myEngine::pEngine->removeObject(viewId);
		Py_RETURN_NONE;
	}
	
	//Navigate
	//Arguments:
	//1. viewId
	//2. url
	static PyObject* engine_navigate(PyObject* self, PyObject* args) {
		Py_ssize_t argCount = PyTuple_Size(args);
		if (argCount < 2) {
			std::string err = "Navigate needs parameters.";
			PyErr_SetString(PyExc_TypeError, err.c_str());
			Py_RETURN_NONE;
		}
		long viewId = PyLong_AsLong(PyTuple_GetItem(args, 0));
		const char* url = nullptr;
		url = PyUnicode_AsUTF8(PyTuple_GetItem(args, 1));
		myEngine::pEngine->Navigate(viewId, url);
		Py_RETURN_NONE;
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
        { "engine_test",  engine_test, METH_VARARGS, "Test method for engine module."},
        { "Path",  engine_path, METH_VARARGS, "Combine MyEngine Root Path with your relative path inside of Root."},
        { "Import",  engine_import, METH_VARARGS, "Import file."},
        { "GetBackground",  engine_getbackground, METH_NOARGS, "Get Background Image Tensor."},
        { "Message",  engine_sendMessage, METH_VARARGS, "Send Message to Host as json string with id and msg fields."},
        { "WebMessage",  engine_sendWebMessage, METH_VARARGS, "Send Message to Web View as json string with id and msg fields."},
        { "AddMainWindow",  engine_addmainwindow, METH_VARARGS, "Open a window."},		
		{ "AddWebView",  engine_addwebview, METH_VARARGS, "Add WebView."},
        { "RemoveWebView",  engine_removewebview, METH_VARARGS, "Remove WebView."},
		{ "Navigate",  engine_navigate, METH_VARARGS, "Navigate WebView."},		
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

