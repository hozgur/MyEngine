#pragma once

namespace My
{
	class Control
	{
	public:
		int borderWidth;
		Color borderColor;
		Color backColor;
		Color foreColor;
		ivec2 pos;
		ivec2 size;
		Control(int x, int y, int width, int height)
		{
			borderWidth = currentTheme::borderWidth;
			borderColor = currentTheme::Border;
			backColor = currentTheme::Background;
			foreColor = currentTheme::Foreground;
			pos = ivec2(x, y);
			size = ivec2(width, height);
		}

		virtual void Draw()
		{
			My::Engine::pEngine->FillRect(pos, size, backColor);
			My::Engine::pEngine->DrawRect(pos, size, borderColor);
		}
	};
}
