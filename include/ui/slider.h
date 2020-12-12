#pragma once
namespace My
{
	class Slider : public Control
	{
	public:
		int thumbWidth;
		Slider(int x, int y, int width, int height):Control(x,y,width,height)
		{			
			thumbWidth = width / 10;
		}

		virtual void Draw() override
		{
			Control::Draw();
		}
	};
}