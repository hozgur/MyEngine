#pragma once
#include "Core.h"
namespace My
{
	template<typename T>
	class image
	{
	public:
		virtual int getWidth() = 0;
		virtual int getHeight() = 0;		
		template<typename F> void getLine(int line, F&& lambda)	// TODO: Read/Write/ReadWrite metodlarýný desteklemeli.
		{
			const std::vector<T> lineData = getline(line);
			lambda(lineData);
			setline(line, lineData);
		};
		virtual image<T>* clone() = 0;
	protected:
		virtual const std::vector<T> getline(int line) = 0;
		virtual void setline(int line, const std::vector<T> &data) = 0;
	};

	//template<typename T>
	//class drawable : public image<T>
	//{
	//public:		
	//	void Draw(int x, int y, T val) {getLine(y, [x, val](const std::vector<T>& data) {data[x] = val; }); }

	//	template<typename F> void getLine(int line, F&& lambda)	// TODO: Read/Write/ReadWrite metodlarýný desteklemeli.
	//	{
	//		const std::vector<T> lineData = getline(line);
	//		lambda(lineData);
	//		setline(line, lineData);
	//	};
	//	virtual const std::vector<T> getline(int line) = 0;
	//	virtual void setline(int line, const std::vector<T>& data) = 0;
	//};

	//template<typename T,int padX = 3,int padY = 3>
	//class padding : public image<T>
	//{
	//	image<T>* image;
	//public:
	//	padding(const image<T>* image) { this->image = image; }
	//	virtual int getWidth()  override { return image->getWidth() + 2 * padX; }
	//	virtual int getHeight() override { return image->getHeight() + 2 * padY; }

	//	
	//	virtual const std::vector<T>& getline(int line) = 0;
	//	virtual void setline(int line, const std::vector<T>& data) = 0;
	//};

}
