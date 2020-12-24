#pragma once
#include "mytensor.h"
#include "structmember.h"
namespace My
{
	namespace Py
	{
		struct pytensor
		{
			PyObject_HEAD
				handle tensor;
			char type;
		};


		static void* pair2tensor(std::pair<char, int> pair,std::vector<int64_t> shape, void* data = nullptr, int64_t byteSize = 0)
		{
			switch (pair.first)
			{
			case 'b':return new tensorImpl<uint8_t>(shape, data, byteSize);
			case 'h':return new tensorImpl<int16_t>(shape, data, byteSize);
			case 'H':return new tensorImpl<uint16_t>(shape, data, byteSize);
			case 'i':return new tensorImpl<int32_t>(shape, data, byteSize);
			case 'I':return new tensorImpl<uint32_t>(shape, data, byteSize);
			case 'L':return new tensorImpl<int64_t>(shape, data, byteSize);
			case 'K':return new tensorImpl<uint64_t>(shape, data, byteSize);
			case 'f':return new tensorImpl<float>(shape, data, byteSize);
			case 'd':return new tensorImpl<double>(shape, data, byteSize);
			case 'x': return nullptr;	// þu anki sistemde custom desteklenmemektedir.
			default:return nullptr;
			}
		}

		static int pytensor_init(pytensor* self, PyObject* args) {
			Py_ssize_t argCount = PyTuple_Size(args);
			const char* typeString = nullptr;
			int itemSize = 0;
			std::vector<int64_t> vshape;
			if (argCount > 0)
			{
				PyObject* shape = PyTuple_GetItem(args, 0);
				if (shape == NULL) { return -1; }

				if(! getTupleList(shape, vshape))
					return -1;				
				
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
			void* tensor = pair2tensor(p, vshape);
			if (tensor == nullptr)
				return -1;
			Engine::pEngine->RemoveMyObject(self->tensor);
			self->tensor = Engine::pEngine->SetMyObject((object*)tensor);
			self->type = p.first;
			//Py_INCREF(self);
			return 0;
		}

		static void pytensor_dealloc(pytensor* self) {
			Engine::pEngine->RemoveMyObject(self->tensor);
			self->tensor = invalidHandle;;
			Py_TYPE(self)->tp_free((PyObject*)self);
		}

		static PyObject* pytensor_str(pytensor* self) {
			std::stringstream buffer;
			buffer << *(tensor<uint8_t>*)Engine::pEngine->GetMyObject(self->tensor);
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
			tensor<uint8_t>* t = (tensor<uint8_t>*)Engine::pEngine->GetMyObject(self->tensor);
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

		static PyMemberDef Custom_Members[] = {
			{"type", T_OBJECT_EX, offsetof(pytensor, type), 0, "Type"},	
			{NULL}  /* Sentinel */
			};
		

		

		
		
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

		static PyObject* fromBuffer(PyObject* self, PyObject* args)
		{
			pytensor* s = (pytensor*)self;
			PyObject* image;
			PyObject* shape;
			if (!PyArg_ParseTuple(args, "OO", &image, &shape))
				return nullptr;
			if (!PyObject_CheckBuffer(image))
			{
				PyErr_SetString(PyExc_TypeError, "buffer is not supported.");
				return nullptr;
			}
			if (!PyTuple_Check(shape))
			{
				PyErr_SetString(PyExc_TypeError, "shape is not validated.");
				return nullptr;
			}
			Py_buffer buffer;
			if (PyObject_GetBuffer(image, &buffer, 0) != 0)
			{
				PyErr_SetString(PyExc_TypeError, "buffer is not supported (2).");
				return nullptr;
			}

			Py_ssize_t shapesize = PyTuple_Size(shape);
			std::vector<int64_t> vshape;
			for (int i = 0; i < shapesize; i++) {
				PyObject* dim = PyTuple_GetItem(shape, i);
				if (!PyNumber_Check(dim)) {
					PyErr_SetString(PyExc_TypeError, "Non-numeric argument on shape.");
					return nullptr;
				}
				PyObject* longNumber = PyNumber_Long(dim);
				vshape.push_back((int64_t)PyLong_AsUnsignedLong(longNumber));
				Py_DECREF(longNumber);
				if (PyErr_Occurred()) { return nullptr; }
			}
			Py_DECREF(shape);

			void* tensor = nullptr;
			char type;
			if (buffer.itemsize == 1)
			{
				tensor = new tensorImpl<uint8_t>(vshape, buffer.buf, buffer.len);
				type = 'b';
			}

			if (buffer.itemsize == 4)
			{
				tensor = new tensorImpl<int>(vshape, buffer.buf, buffer.len);
				type = 'i';
			}
			if (tensor == nullptr)
			{
				PyErr_SetString(PyExc_TypeError, "itemSize is not supported.");
				return nullptr;
			}
			pytensor* tens = (pytensor*)PyObject_CallObject((PyObject*)&pytensorType, 0);

			//Engine::pEngine->RemoveMyObject(s->tensor);
			tens->tensor = Engine::pEngine->SetMyObject((object*)tensor);			
			tens->type = type;
			return (PyObject*)tens;
		}

		static PyMethodDef Custom_Methods[] = {
			{"fromBuffer", (PyCFunction)fromBuffer, METH_VARARGS,
			 "Create Tensor from PIL Image"
			},
			{NULL}  /* Sentinel */
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
			pytensorType.tp_members = Custom_Members;
			pytensorType.tp_methods = Custom_Methods;
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