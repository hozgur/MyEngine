#pragma once

class myView : public myObject
{
public:
	myView* parent = nullptr;
	virtual bool SetSize(int width, int height) = 0;
	virtual bool SetPosition(int x, int y) = 0;
	virtual bool GetSize(int& width, int& height) = 0;
	virtual bool GetPosition(int& x, int& y) = 0;
};