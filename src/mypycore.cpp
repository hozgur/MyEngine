#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "my.h"
#include "mypycore.h"
namespace myPy
{	
	template<>
	bool getTupleList<long>(PyObject* tuple, std::vector<long>& list)
	{
		if (!PyTuple_Check(tuple))
		{
			PyErr_SetString(PyExc_TypeError, "shape must be list.");
			return false;
		}
		Py_ssize_t size = PyTuple_Size(tuple);

		for (int i = 0; i < size; i++) {
			PyObject* item = PyTuple_GetItem(tuple, i);
			if (!PyNumber_Check(item)) {
				PyErr_SetString(PyExc_TypeError, "Non-numeric argument on list.");
				return false;
			}
			list.push_back(PyLong_AsUnsignedLong(item));
			if (PyErr_Occurred()) { return false; }
		}
		return true;
	}

	template<>
	bool getTupleList<int64_t>(PyObject* tuple, std::vector<int64_t>& list)
	{
		if (!PyTuple_Check(tuple))
		{
			PyErr_SetString(PyExc_TypeError, "shape must be list.");
			return false;
		}
		Py_ssize_t size = PyTuple_Size(tuple);

		for (int i = 0; i < size; i++) {
			PyObject* item = PyTuple_GetItem(tuple, i);
			if (!PyNumber_Check(item)) {
				PyErr_SetString(PyExc_TypeError, "Non-numeric argument on list.");
				return false;
			}
			list.push_back((int64_t)PyLong_AsUnsignedLong(item));
			if (PyErr_Occurred()) { return false; }
		}
		return true;
	}
}
	
