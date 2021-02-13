#pragma once
#include "my.h"
#include "molecular/molecular2.h"
using namespace My;

template<typename T>
class container
{
public:
	typedef int size_type;
	class iterator
	{
	public:
		typedef iterator self_type;
		typedef T value_type;
		typedef T& reference;
		typedef T* pointer;
		typedef std::forward_iterator_tag iterator_category;
		typedef int difference_type;
		iterator(pointer ptr) : ptr_(ptr) { }
		self_type operator++() { self_type i = *this; ptr_++; return i; }
		self_type operator++(int junk) { ptr_++; return *this; }
		reference operator*() { return *ptr_; }
		pointer operator->() { return ptr_; }
		bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
		bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
	private:
		pointer ptr_;
	};

	container(size_type size) :size(size), data(new T[size]) { for (size_type a = 0; a < size; a++) data[a] = a; }
	
	iterator begin() { return iterator(data); }
	iterator end() { return iterator(data + size); }

private:
	T* data;
	size_type size;
};


// Range iterator sample
class range
{
public:
	range(int start,int end) :start(start),current(start),_end(end) {}

	class iterator
	{
	public:
		iterator(int current):current(current){}
		
		bool operator!=(const iterator& rhs) { return current != rhs.current; }
		iterator operator++() { iterator i = *this; current++; return i; }
		int& operator*() { return current; }
		int current;
	};

	iterator begin() { return iterator(start); }
	iterator end() { return iterator(_end); }

	int current;
	int start;
	int _end;
};


class MyEngine : public My::Engine
{
public:

	const int width = 400;
	const int height = 300;
	grid<int> g;
	MyEngine(const char* path) :My::Engine(path),g(width,height){}
	void InitDots()
	{
		for (int a = 0; a < 10; a++)
			g.add(a * 3., a * 3., pixel<int>(My::Color::Random()));
	}
	bool OnStart() override
	{
		//AddWindow(width, height, 1, 1);
		InitDots();

		
		return true;
	}
	void MoveDot(int nDot)
	{

	}
	
	void OnDraw() override
	{
	
	}

	void OnUpdate() override
	{
		
	}

	void OnExit() override
	{

	}
};