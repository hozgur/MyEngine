#include "my.h"


// Inherited via myView
bool myEngine::SetSize(int width, int height) {
	
	return false;
}
bool myEngine::SetPosition(int x, int y) {
	
	return false;
}
bool myEngine::GetSize(int& width, int& height) {
	width = clientWidth;
	height = clientHeight;
	return true;
}
bool myEngine::GetPosition(int& x, int& y) {
	return false;
}

bool myEngine::SetAnchors(myAnchor anchors)
{
	return false;
}
