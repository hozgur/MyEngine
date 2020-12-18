#include "my.h"

namespace My
{
    void Engine::Clear(Color c,handle h)
    {
        if(h < 0)
            pPlatform->ClearBackground(c);
        else
        {

        }

    }

    void Engine::DrawHLine(int x1, int x2, int y, const Color& c)
    {
        if (x1 < 0) x1 = 0;
        if (x2 < 0) return;
        if (x1 >= clientWidth) return;
        if (x2 >= clientWidth) x2 = clientWidth-1;
        if (y < 0) return;
        if (y >= clientHeight) return;

        Color* p = GetLinePointer(y);
        for (int x = x1; x <= x2; x++)
            p[x] = c;
    }

    void Engine::DrawVLine(int y1, int y2, int x, const Color& c)
    {
        if (y1 < 0) y1 = 0;
        if (y2 < 0) return;
        if (y1 >= clientHeight) return;
        if (y2 >= clientHeight) y2 = clientHeight-1;
        if (x < 0) return;
        if (x >= clientWidth) return;

        for (int y = y1; y <= y2; y++)
            GetLinePointer(y)[x] = c;
    }

    void Engine::FillRect(ivec2 pos, ivec2 size, const Color& c)
    {
        for (int y = pos.y(); y < pos.y() + size.y(); y++)
        {
            DrawHLine(pos.x(), pos.x() + size.x()-1, y, c);
        }
    }

    void Engine::DrawRect(ivec2 pos, ivec2 size, const Color& c)
    {
        DrawHLine(pos.x(), pos.x() + size.x()-1, pos.y(), c);
        DrawHLine(pos.x(), pos.x() + size.x()-1, pos.y() + size.y() - 1, c);
        DrawVLine(pos.y() + 1, pos.y() + size.y()-1, pos.x(), c);
        DrawVLine(pos.y() + 1, pos.y() + size.y()-1, pos.x() + size.x() - 1, c);
    }

    void Engine::FillCircle(ivec2 pos, int radius, const Color& c)
    {
        int x = pos.x();
        int y = pos.y();
        if (radius > 0)
        {
            int x0 = 0;
            int y0 = radius;
            int d = 3 - 2 * radius;

            while (y0 >= x0)
            {
                DrawHLine(x - y0, x + y0, y - x0, c);
                if (x0 > 0)	DrawHLine(x - y0, x + y0, y + x0, c);

                if (d < 0)
                    d += 4 * x0++ + 6;
                else
                {
                    if (x0 != y0)
                    {
                        DrawHLine(x - x0, x + x0, y - y0, c);
                        DrawHLine(x - x0, x + x0, y + y0, c);
                    }
                    d += 4 * (x0++ - y0--) + 10;
                }
            }
        }
        else
        {
            if ((x >= 0) && (y >= 0) && (x <= clientWidth) && (y <= clientHeight))
                Pixel(x, y) = c;
        }
    }

    void Engine::DrawLine(int32_t x1, int32_t y1, int32_t x2, int32_t y2, Color p, uint32_t pattern)
    {
        int x, y, dx, dy, dx1, dy1, px, py, xe, ye, i;
        dx = x2 - x1; dy = y2 - y1;

        auto rol = [&](void) { pattern = (pattern << 1) | (pattern >> 31); return pattern & 1; };
        auto Draw = [this](int x, int y, const Color& c) {Pixel(x, y) = c; };
        // straight lines idea by gurkanctn
        if (dx == 0) // Line is vertical
        {
            if (y2 < y1) std::swap(y1, y2);
            for (y = y1; y <= y2; y++) if (rol()) Draw(x1, y, p);
            return;
        }

        if (dy == 0) // Line is horizontal
        {
            if (x2 < x1) std::swap(x1, x2);
            for (x = x1; x <= x2; x++) if (rol()) Draw(x, y1, p);
            return;
        }

        // Line is Funk-aye
        dx1 = abs(dx); dy1 = abs(dy);
        px = 2 * dy1 - dx1;	py = 2 * dx1 - dy1;
        if (dy1 <= dx1)
        {
            if (dx >= 0)
            {
                x = x1; y = y1; xe = x2;
            }
            else
            {
                x = x2; y = y2; xe = x1;
            }

            if (rol()) Draw(x, y, p);

            for (i = 0; x < xe; i++)
            {
                x = x + 1;
                if (px < 0)
                    px = px + 2 * dy1;
                else
                {
                    if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) y = y + 1; else y = y - 1;
                    px = px + 2 * (dy1 - dx1);
                }
                if (rol()) Draw(x, y, p);
            }
        }
        else
        {
            if (dy >= 0)
            {
                x = x1; y = y1; ye = y2;
            }
            else
            {
                x = x2; y = y2; ye = y1;
            }

            if (rol()) Draw(x, y, p);

            for (i = 0; y < ye; i++)
            {
                y = y + 1;
                if (py <= 0)
                    py = py + 2 * dx1;
                else
                {
                    if ((dx < 0 && dy < 0) || (dx > 0 && dy > 0)) x = x + 1; else x = x - 1;
                    py = py + 2 * (dx1 - dy1);
                }
                if (rol()) Draw(x, y, p);
            }
        }
    }

    void Engine::DrawLine(ivec2 p1, ivec2 p2, Color p, uint32_t pattern)
    {
        DrawLine(p1.x(), p1.y(), p2.x(), p2.y(), p, pattern);
    }
    void Engine::DrawText(int x, int y, std::string text, int fontHeight)
    {
    }    
}
