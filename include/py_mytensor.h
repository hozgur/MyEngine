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

		/*static void* pair2mytensor(std::pair<char, int> pair)
		{
			switch (pair.first)
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
		}*/

		/*static int pytensor_init(pytensor* self, PyObject* args, PyObject* kwds) {
			
			if (self->tensor != nullptr)
				delete self->tensor;
			int size = 0;
			int itemsize = 0;
			const char* datatype = nullptr;
			static char sizestr[5] = "size";
			static char typestr[5] = "type";
			static char itemsizestr[9] = "itemsize";
			static char* kwlist[] = { sizestr, typestr, itemsizestr, nullptr };
			if (!PyArg_ParseTupleAndKeywords(args, kwds, "i|si", kwlist, &size, &datatype, &itemsize))
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
		}*/
	}
}