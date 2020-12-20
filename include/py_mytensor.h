#pragma once
#include "mytensor.h"
namespace My
{
	namespace Py
	{
		struct pytensor
		{
			PyObject_HEAD
			void* tensor;
			char type;
		};

		static char types[] = "b h H i I L K f d x ";

		static std::pair<char, int> str2pair(const char* typestring)
		{
			std::string str(typestring);
			if (str == "byte") return   { 'b',1 };
			if (str == "short") return	{ 'h',2 };
			if (str == "ushort") return { 'H',2 };
			if (str == "int") return	{ 'i',4 };
			if (str == "uint") return	{ 'I',4 };
			if (str == "long") return	{ 'L',8 };
			if (str == "ulong") return	{ 'K',8 };
			if (str == "float") return	{ 'f',4 };
			if (str == "double") return { 'd',8 };
			if (str == "custom") return { 'x',0 };

			return { 'e',-1 };
		}

		static std::string char2str(char  type)
		{
			switch (type)
			{
			case 'b':return "byte";
			case 'h':return "short";
			case 'H':return "ushort";
			case 'i':return "int";
			case 'I':return "uint";
			case 'L':return "long";
			case 'K':return "ulong";
			case 'f':return "float";
			case 'd':return "double";
			case 'x':return "custom";
			default:return "undefined";
			}
		}

		static int char2itemsize(char  type)
		{
			switch (type)
			{
			case 'b':return 1;
			case 'h':return 2;
			case 'H':return 2;
			case 'i':return 4;
			case 'I':return 4;
			case 'L':return 8;
			case 'K':return 8;
			case 'f':return 4;
			case 'd':return 8;
			case 'x':return 0;
			default:return 0;
			}
		}

		static void* pair2tensor(std::pair<char, int> pair,std::vector<int64_t> shape)
		{
			switch (pair.first)
			{
			case 'b':return new mytensorImpl<uint8_t>(shape);
			case 'h':return new mytensorImpl<int16_t>(shape);
			case 'H':return new mytensorImpl<uint16_t>(shape);
			case 'i':return new mytensorImpl<int32_t>(shape);
			case 'I':return new mytensorImpl<uint32_t>(shape);
			case 'L':return new mytensorImpl<int64_t>(shape);
			case 'K':return new mytensorImpl<uint64_t>(shape);
			case 'f':return new mytensorImpl<float>(shape);
			case 'd':return new mytensorImpl<double>(shape);
			case 'x': return nullptr;	// þu anki sistemde custom desteklenmemektedir.
			default:return nullptr;
			}
		}

		static int pytensor_init(pytensor* self, PyObject* args) {
			
			if (self->tensor != nullptr)
				delete self->tensor;
			Py_ssize_t argCount = PyTuple_Size(args);
			const char* typeString = nullptr;
			int itemSize = 0;
			std::vector<int> vshape;
			if (argCount > 0)
			{
				PyObject* shape = PyTuple_GetItem(args, 0);
				if (shape == NULL) { return -1; }

				if (!PyTuple_Check(shape))
				{
					PyErr_SetString(PyExc_TypeError, "shape must be list.");
					return -1;
				}
				Py_ssize_t shapesize = PyTuple_Size(shape);

				for (int i = 0; i < shapesize; i++) {
					PyObject* dim = PyTuple_GetItem(shape, i);
					if (!PyNumber_Check(dim)) {
						PyErr_SetString(PyExc_TypeError, "Non-numeric argument on shape.");
						return -1;
					}
					PyObject* longNumber = PyNumber_Long(dim);
					vshape.push_back((int)PyLong_AsUnsignedLong(longNumber));
					Py_DECREF(longNumber);
					if (PyErr_Occurred()) { return -1; }
				}
				Py_DECREF(shape);

				if (argCount > 1)
				{
					PyObject* typeStr = PyTuple_GetItem(args, 1);
					Py_ssize_t size;
					typeString = PyUnicode_AsUTF8AndSize(typeStr, &size);
					Py_DECREF(typeStr);
				}

				if (argCount > 2)
				{
					PyObject* _itemSize = PyTuple_GetItem(args, 2);
					itemSize = PyLong_AsLong(_itemSize);
					Py_DECREF(_itemSize);
				}
			}
			
			std::pair<char, int> p = { 'i' , 4 };
			if (typeString != nullptr)
				p = str2pair(typeString);
			itemSize = p.second;
			if (itemSize < 0)
			{
				PyErr_SetString(PyExc_ValueError, "Undefined Type.");
				return -1;
			}
			if (p.first != 'x')
				itemSize = p.second;
			std::vector<int64_t> shape64(vshape.begin(), vshape.end());
			void* tensor = pair2tensor(p, shape64);
			if (tensor == nullptr)
				return -1;
			self->tensor = tensor;
			self->type = p.first;
			return 0;
		}

		static void pytensor_dealloc(pytensor* self) {
			delete self->tensor;
			self->tensor = nullptr;
			Py_TYPE(self)->tp_free((PyObject*)self);
		}

		static PyObject* pytensor_str(pytensor* self) {
			std::stringstream buffer;
			buffer << *(mytensor<uint8_t>*)self->tensor;
			return PyUnicode_FromString(buffer.str().c_str());
		}

		static int pytensor_getbuffer(PyObject* obj, Py_buffer* view, int flags)
		{
			if (view == nullptr)
			{
				PyErr_SetString(PyExc_ValueError, "NULL View in get buffer.");
				return -1;
			}
			pytensor* self = (pytensor*)obj;
			mytensor<uint8_t>* t = (mytensor<uint8_t>*)self->tensor;
			if(t->getMinDepth() != 0)
			{
				PyErr_SetString(PyExc_ValueError, "MyTensor is too complex for Python Buffer.");
				return -1;
			}
			view->obj = (PyObject*)self;
			view->buf = t->getData(0,0);
			ssize_t  len = 1;
			int itemsize = char2itemsize(self->type);
			if (t->shape().size() > 0)
			{
				len = t->shape()[0] * t->strides()[0];
			}
			else
			{
				len = itemsize;
			}

			view->len = len;
			view->readonly = 0;
			view->itemsize = itemsize;
			std::string s(1, self->type);
			view->format = nullptr;
			char fmtstr[2];
			memcpy(fmtstr, s.c_str(), 1);
			fmtstr[1] = 0;
			view->format = fmtstr;
			view->ndim = (int) t->shape().size();
			view->shape = (int64_t*) t->shape().data();
			view->strides = (int64_t*) t->strides().data();
			view->suboffsets = nullptr;
			view->internal = nullptr;
			Py_INCREF(self);
			return 0;
		}

		static PyBufferProcs pytensor_as_buffer = {
			(getbufferproc)pytensor_getbuffer,
			(releasebufferproc)0
		};

		static PyTypeObject pytensorType = {
			PyVarObject_HEAD_INIT(NULL, 0)
			"mytensor.MyTensor",			/* tp_name */
			sizeof(pytensor),				/* tp_basicsize */
			0,								/* tp_itemsize */
			(destructor)pytensor_dealloc,	/* tp_dealloc */
			0,								/* tp_print */
			0,								/* tp_getattr */
			0,								/* tp_setattr */
			0,								/* tp_reserved */
			(reprfunc)pytensor_str,			/* tp_repr */
			0,								/* tp_as_number */
			0,								/* tp_as_sequence */
			0,								/* tp_as_mapping */
			0,								/* tp_hash  */
			0,								/* tp_call */
			(reprfunc)pytensor_str,			/* tp_str */
			0,								/* tp_getattro */
			0,								/* tp_setattro */
			&pytensor_as_buffer,			/* tp_as_buffer */
			Py_TPFLAGS_DEFAULT,				/* tp_flags */
			"MyTensor object",				/* tp_doc */
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
			(initproc)pytensor_init,			/* tp_init */
		};

		static PyModuleDef pytensor_module = {
			PyModuleDef_HEAD_INIT,
			"mytensor",
			"Extension type for MyTensor object.",
			-1,
			NULL, NULL, NULL, NULL, NULL
		};

		PyMODINIT_FUNC PyInit_pytensor(void)
		{
			PyObject* m;
			pytensorType.tp_new = PyType_GenericNew;
			if (PyType_Ready(&pytensorType) < 0)
				return NULL;

			m = PyModule_Create(&pytensor_module);
			if (m == NULL)
				return NULL;

			Py_INCREF(&pytensorType);
			PyModule_AddObject(m, "MyTensor", (PyObject*)&pytensorType);
			return m;
		}
		bool initTensorModule()
		{
			if (PyImport_AppendInittab("mytensor", PyInit_pytensor) == -1) {
				debug << "Error: could not extend in-built modules table\n";
				return false;
			}
			return true;
		}
	}
}