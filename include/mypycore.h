#pragma once
namespace myPy
{
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
	
	template<typename T>
	bool getTupleList(PyObject* objTuple, std::vector<T>& list);
}

