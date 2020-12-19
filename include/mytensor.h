#pragma once
#include "My.h"

namespace My
{
	template<typename T>
	class mytensor
	{
	public:
		std::vector<int64_t> shape() = 0;
		std::vector<int64_t> strides() = 0;
		T* getData(int depth) = 0;
		void setData(int depth, const T* data) = 0;
	};
}