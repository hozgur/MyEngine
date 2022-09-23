#pragma once

enum myAnchor {
	myAnchorNone = 0,
	myAnchorLeft = 1,
	myAnchorRight = 2,
	myAnchorTop = 4,
	myAnchorBottom = 8,
	myAnchorAll = 15
};


class myView : public myObject
{
public:
	myView* parent = nullptr;
	myAnchor anchor = myAnchorNone;
	int marginLeft = 0;
	int marginTop = 0;
	int marginRight = 0;
	int marginBottom = 0;
	
	virtual bool SetAnchors(myAnchor anchors) = 0;
	
	virtual bool SetSize(int width, int height) = 0;
	virtual bool SetPosition(int x, int y) = 0;
	virtual bool GetSize(int& width, int& height) = 0;
	virtual bool GetPosition(int& x, int& y) = 0;

};