#pragma once
#include "My.h"

namespace My
{
	template<typename T>
	class tensor : public object
	{
	public:
		virtual const std::vector<int64_t>& shape() const = 0;
		virtual const std::vector<int64_t>& strides() const = 0;
		virtual T* getData() const = 0;
		virtual bool canUseReadforWrite() const = 0;
		virtual void setData(const T* data, int dataSize) const = 0;		
	};

	template<typename T>
	class tensorImpl : public tensor<T>
	{
	public:
		std::vector<int64_t> _shape;
		std::vector<int64_t> _strides;
		T* pData = nullptr;
		bool deleteData = true;

		tensorImpl(const tensorImpl&) = delete;
		tensorImpl& operator=(const tensorImpl&) = delete;

		tensorImpl(std::vector<int64_t> shape, void* data = nullptr, int64_t byteSize = 0) {
			int64_t size = 1; for (int64_t d : shape) size *= d;			
			this->_shape = shape;
			if ((data != nullptr) && (byteSize == (size * sizeof(T))))
			{
				pData = (T*) data;
				deleteData = false;
			}
			else
			{
				pData = new T[size];
			}
		}
		~tensorImpl() {
			if(deleteData)
				delete pData;
		}
		// mytensor impl
		const std::vector<int64_t>& shape() const override { return _shape; }
		const std::vector<int64_t>& strides() const override{ return _strides; }
		T* getData() const override {
			return pData;
			}
		virtual bool canUseReadforWrite() const override { return true; }
		void setData(const T* data, int dataSize) const override{}
	};

	//class imagetensor : public image<Color>
	//{
	//public:		
	//	tensor<uint8_t>* sourceTensor = nullptr;

	//	imagetensor(tensor<uint8_t>* tensor);
	//	
	//	// Image Methods
	//	virtual int getWidth() const override { return sourceTensor->shape()[2]; }
	//	virtual int getHeight() const override { return sourceTensor->shape()[1]; }
	//	virtual Interleave getInterleave() const override { return Interleave::interleave_planar; }
	//	virtual Color* readLine(int line) const override { return (Color*)sourceTensor->getData(1, line); }
	//	virtual bool canUseReadforWrite() const override { return sourceTensor->canUseReadforWrite(); }
	//	virtual void writeLine(int line, const Color* data, int bytecount) override { sourceTensor->setData(1, line, data, bytecount); }
	//	virtual void draw(image<Color>* source, int x, int y, int dW, int dH, int sX = 0, int sY = 0, int sW = -1, int sH = -1, Interpolation interpolation = Interpolation::interpolation_default) const override{
	//		
	//	}
	//};


	template<typename T>
	std::ostream& operator<<(std::ostream& os, const tensor<T>& t)
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
