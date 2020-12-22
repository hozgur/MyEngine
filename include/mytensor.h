#pragma once
#include "My.h"

namespace My
{
	template<typename T>
	class mytensor : public object
	{
	public:
		virtual const std::vector<int64_t>& shape() const = 0;
		virtual const std::vector<int64_t>& strides() const = 0;
		virtual int getMinDepth() const = 0;
		virtual T* getData(int depth, int index) const = 0;
		virtual void setData(int depth, int index, const T* data, int dataSize) const = 0;
	};

	template<typename T>
	class mytensorImpl : public mytensor<T>
	{
	public:
		std::vector<int64_t> _shape;
		std::vector<int64_t> _strides;
		T* pData = nullptr;

		mytensorImpl(const mytensorImpl&) = delete;
		mytensorImpl& operator=(const mytensorImpl&) = delete;

		mytensorImpl(std::vector<int64_t> shape, void* data = nullptr, int64_t byteSize = 0) {
			int64_t size = 1; for (int64_t d : shape) size *= d;
			for (int i = 0; i < shape.size(); i++)
			{
				int s = sizeof(T);
				for (int j = i+1; j < shape.size(); j++) s *= (int)shape[j];
				_strides.push_back(s);
			}
			this->_shape = shape;
			if ((data != nullptr) && (byteSize == (size * sizeof(T))))
			{
				pData = (T*) data;
			}
			else
			{
				pData = new T[size];
			}
		}
		~mytensorImpl() {
			delete pData;
		}
		// mytensor impl
		const std::vector<int64_t>& shape() const override { return _shape; };
		const std::vector<int64_t>& strides() const override{ return _strides; };
		T* getData(int depth, int index) const override {
			if (depth <= 0) return pData;
			return (T*)((uint8_t*)pData + _strides[depth -1 ] * index);
			}
		void setData(int depth, int index, const T* data, int dataSize) const override{}
		virtual int getMinDepth() const override { return 0; }
	};	
	template<typename T>
	std::ostream& operator<<(std::ostream& os, const mytensor<T>& t)
	{
		auto shape = t.shape();
		os << "Tensor<"; 
		if (shape.size() == 0)
			os << sizeof(T) << ">\n";
		else
		{
			os << t.strides().back();
				os << "> : Shape: [ ";
			for (int i = 0; i < shape.size(); i++)
				os << shape[i] << " ";
			os << "]\n";
		}
		return os;
	}
}
