#pragma once
namespace myPy
{
	struct myarray {
		void* arr;
		int64_t size;
		int itemSize;
		// type  Python Representation
		//		 
		// b - > "byte"
		// h - > "short" (16 bit)
		// H - > "ushort"
		// i - > "int" (32bit)
		// I - > "uint"
		// L - > "long" (64 bit)
		// K - > "ulong" (64 bit)
		// f - > "float"
		// d - > "double"
		// x - > "custom"
		char type;
		static void init_array(myarray* a, int64_t size, char type, int itemsize) {
			if (size > 0)
				a->arr = malloc(size * itemsize);
			else
				a->arr = nullptr;
			a->size = size;
			a->itemSize = itemsize;
			a->type = type;
		}

		static void del_array(myarray* a) {
			free(a->arr);
			a->arr = nullptr;
		}
	};

	bool initArrayModule();
}


namespace myPy
{		
	typedef struct {
		PyObject_HEAD
			myarray arr;
	}pymyarray;

	static int pymyarray_init(pymyarray* self, PyObject* args, PyObject* kwds) {
		/*int len = strlen(types);
		for (int a = 1; a < len; a += 2)
			types[a] = 0;*/
		if (self->arr.arr != nullptr)
			myarray::del_array(&self->arr);
		int size = 0;
		int itemsize = 0;
		const char* datatype = nullptr;
		static char sizestr[5] = "size";
		static char typestr[5] = "type";
		static char itemsizestr[9] = "itemsize";
		static char* kwlist[] = { sizestr, typestr, itemsizestr, nullptr };
		if (!PyArg_ParseTupleAndKeywords(args, kwds, "|isi", kwlist, &size, &datatype, &itemsize))
			return -1;
		if (size < 0) size = 0;
		std::pair<char, int> p = { 'i' , 4 };
		if (datatype != nullptr)
			p = str2pair(datatype);
		itemsize = p.second;
		if (itemsize < 0)
		{
			PyErr_SetString(PyExc_ValueError, "Undefined Type.");
			return -1;
		}
		if (p.first != 'x')
			itemsize = p.second;

		myarray::init_array(&self->arr, size, p.first, itemsize);
		return 0;
	}

	static void pymyarray_dealloc(pymyarray* self) {
		myarray::del_array(&self->arr);
		Py_TYPE(self)->tp_free((PyObject*)self);
	}

	static PyObject* pymyarray_str(pymyarray* self) {
		std::string str = "<MyArray Size = " + std::to_string(self->arr.size) + " Item Type = " + char2str(self->arr.type) + " Item Size = " + std::to_string(self->arr.itemSize) + ">";
		return PyUnicode_FromString(str.c_str());
	}

	static int pymyarray_getbuffer(PyObject* obj, Py_buffer* view, int flags)
	{
		if (view == nullptr)
		{
			PyErr_SetString(PyExc_ValueError, "NULL View in get buffer.");
			return -1;
		}
		pymyarray* self = (pymyarray*)obj;
		debug << "getbuffer";
		view->obj = (PyObject*)self;
		view->buf = self->arr.arr;
		view->len = self->arr.size * self->arr.itemSize;
		view->readonly = 0;
		view->itemsize = self->arr.itemSize;
		std::string s(1, self->arr.type);
		view->format = nullptr;
		char fmtstr[2];
		memcpy(fmtstr, s.c_str(), 1);
		fmtstr[1] = 0;
		if (s != "x")
			view->format = fmtstr;
		view->ndim = 1;
		view->shape = &self->arr.size;
		view->strides = &view->itemsize;
		view->suboffsets = nullptr;
		view->internal = nullptr;
		Py_INCREF(self);
		return 0;
	}

	static PyBufferProcs pymyarray_as_buffer = {
		(getbufferproc)pymyarray_getbuffer,
		(releasebufferproc)0
	};

	static PyTypeObject pymyarrayType = {
		PyVarObject_HEAD_INIT(NULL, 0)
		"pymyarray.PyMyArray",			/* tp_name */
		sizeof(pymyarray),				/* tp_basicsize */
		0,								/* tp_itemsize */
		(destructor)pymyarray_dealloc,	/* tp_dealloc */
		0,								/* tp_print */
		0,								/* tp_getattr */
		0,								/* tp_setattr */
		0,								/* tp_reserved */
		(reprfunc)pymyarray_str,			/* tp_repr */
		0,								/* tp_as_number */
		0,								/* tp_as_sequence */
		0,								/* tp_as_mapping */
		0,								/* tp_hash  */
		0,								/* tp_call */
		(reprfunc)pymyarray_str,			/* tp_str */
		0,								/* tp_getattro */
		0,								/* tp_setattro */
		&pymyarray_as_buffer,			/* tp_as_buffer */
		Py_TPFLAGS_DEFAULT,				/* tp_flags */
		"PyMyArray object",				/* tp_doc */
		0,								/* tp_traverse */
		0,								/* tp_clear */
		0,								/* tp_richcompare */
		0,								/* tp_weaklistoffset */
		0,								/* tp_iter */
		0,								/* tp_iternext */
		0,								/* tp_methods */
		0,								/* tp_members */
		0,								/* tp_getset */
		0,								/* tp_base */
		0,								/* tp_dict */
		0,								/* tp_descr_get */
		0,								/* tp_descr_set */
		0,								/* tp_dictoffset */
		(initproc)pymyarray_init,			/* tp_init */
	};

	static PyModuleDef pymyarray_module = {
		PyModuleDef_HEAD_INIT,
		"PyMyArray",
		"Extension type for myarray object.",
		-1,
		NULL, NULL, NULL, NULL, NULL
	};

	PyMODINIT_FUNC PyInit_pymyarray(void)
	{
		PyObject* m;
		pymyarrayType.tp_new = PyType_GenericNew;
		if (PyType_Ready(&pymyarrayType) < 0)
			return NULL;

		m = PyModule_Create(&pymyarray_module);
		if (m == NULL)
			return NULL;

		Py_INCREF(&pymyarrayType);
		PyModule_AddObject(m, "PyMyArray", (PyObject*)&pymyarrayType);
		return m;
	}
	bool initArrayModule()
	{
		if (PyImport_AppendInittab("pymyarray", PyInit_pymyarray) == -1) {
			debug << "Error: could not extend in-built modules table\n";
			return false;
		}
		return true;
	}
}