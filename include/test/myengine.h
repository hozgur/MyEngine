#pragma once
#define I_want_to_create_my_custom_main
#include "my.h"
using namespace My;
const float pi = (float)std::atan(1) * 4;

const int lineCount = 500;
struct line
{
    line(float x, float y, float dx, float dy, myColor color)
    {
        this->x = x;
        this->y = y;
        this->dx = dx;
        this->dy = dy;
        this->color = color;
    }
    float x, y, dx, dy;
    myColor color;


};

class MyEngine : public myEngine
{
public:
    std::vector<line> lines;
    std::string title = "My Engine";
    MyEngine() :myEngine()
    {
    
    }

    bool OnStart() override
    {
        for (int i = 0; i < lineCount; i++)
        {
            lines.push_back(line((float)(rand() % clientWidth / 10), (float)(rand() % clientHeight / 1), (rand() % 500 + 1) / 500.f, 0/*(rand() % 500 + 1) / 5000.f*/, myColor(rand() % 256, rand() % 256, rand() % 256)));
        }
        return true;
    }
    void OnDraw() override
    {
        SetWindowTitle(title);
    }
    void moveLine(int index)
    {
        line& l = lines[index];
        l.x += l.dx;
        l.y += l.dy;
        if (l.x > (clientWidth-1) || l.x < 0)
        {
            l.x -= l.dx;
            l.dx = -l.dx;
        }
        if (l.y > (clientHeight-1) || l.y < 0)
        {
            l.y -= l.dy;
            l.dy = -l.dy;
        }
    }
    bool OnUpdate() override
    {
        myStopWatch s;
        s.Start();
        fade2();
        s.Stop();
        title = std::to_string(s.GetDuration());
        myColor* p = GetLinePointer(0);
        for (int i = 0; i < lineCount; i++)
        {
            int y = (int)lines[i].y;
            int x = (int)lines[i].x;
            p[y * clientWidth + x] = lines[i].color;
            moveLine(i);
        }
        return true;
    }

    void fade()
    {
//#pragma omp parallel for
        for (int y = 0; y < clientHeight; y++)
        {
            myColor* p = GetLinePointer(y);
            for (int x = 0; x < clientWidth; x++)
            {
                p[x].r = 254 * p[x].r / 255;
                p[x].g = 254 * p[x].g / 255;
                p[x].b = 254 * p[x].b / 255;
                
            }
        }
    }

    void fade2()
    {
        ForEachPixel([this](int x, int y, myColor &c) {            
            c = myColor(254*c.r / 255, 254*c.g / 255, 254*c.b / 255);            
            });
    }
    
};
