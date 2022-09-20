#pragma once
#include <iomanip>
#include "my.h"

const int dotCountX = 9;
const int dotCountY = 9;
const float gap = 6;
float distance = 5;
float distance2 = distance * distance;
const int blockSize = 2;
const int scale = 4;
vec2 g(0.f,0.0000f);
using namespace My;
enum Type{Simple,Wall};
struct dot
{
    dot(vec2 pos, vec2 vel, myColor color,int type = Simple)
    {
        this->pos = pos;
        this->vel = vel;        
        this->color = color;
        this->type = type;
    }
    vec2 pos;
    vec2 vel;
    myColor color;
    int type;
};


class MyEngine : public myEngine
{
public:
    std::vector<dot> dots;
    std::string title = "My Engine";
    int width;
    int height;
    MyEngine(const char* argc) :myEngine(argc),width(0),height(0)
    {
        
    }
    
    void InitDots()
    {
        float x1 = 0;
        dots.push_back(dot({ 0,0}, { 0,0 }, myColor::White));
        float y1 = 0.8660254 * gap;
        for (int y = 0; y < dotCountY; y++)
        {
            x1 = x1 + gap/2;
            for (int x = 0; x < dotCountX; x++)
            {
                dots.push_back(dot({ 20 + x*gap + x1 ,10 + y* y1  }, { 0,0 }, myColor::Random()));
            }
            if (x1 == gap) x1 = 0;
        }
        for (int x = 0; x < 200; x += gap)
        {
            dots.push_back(dot({ 40 + x ,80+x*0}, { 0,0 }, myColor::Blue,Wall));
        }
    }

    // DÝKKAT! Kuvvet uygulama ile konum deðiþtirmeyi birbirinden ayýrmam lazým. Hareket mekaniði tekrar düþünülecek.
    void MoveDot(uint32_t nDot)
    {
        dot &d = dots[nDot];
        if (d.type == Wall)
            return;
        d.vel += g;
        d.vel *= 0.95;
        vec2 f(0, 0);
        for (uint32_t i = 0; i < dots.size(); i++)
        {
            if (i != nDot)
            {
                dot& d1 = dots[i];
                if (dist(d.pos, d1.pos) <= gap)
                {
                    float a = -attraction(d.pos, d1.pos);
                    vec2 dv = (d.pos - d1.pos);
                    float n = dv.norm();
                    dv = dv / n;
                    dv = dv * a/1050;
                    f += dv;
                   //  DrawLine((scale * d.pos).cast<int>(),(scale * d1.pos).cast<int>(), myColor::Gray, 0xAAAAAAAA);
                }                                
            }
        }
        
        d.vel += f; 
        /*if (d.vel.norm() > 0.05)
        {
            d.vel *= 0.05 / d.vel.norm();
            debug << std::setprecision(3) << d.vel.norm() << "\n";
        }*/
            
        d.pos.x() = d.pos.x() + d.vel.x();
        d.pos.y() = d.pos.y() + d.vel.y();        
        if (d.pos.x() < 0)
            d.pos.x() = (float)(width-1-blockSize);
        else
            if (d.pos.x() > (width - 1 - blockSize)) d.pos.x() = 0.f;

        if (d.pos.y() < 0) 
            d.pos.y() = (float)(height-1 - blockSize);
        else
            if (d.pos.y() > (height - 1 - blockSize)) d.pos.y() = 0.f;
        
        
    }

    bool OnStart() override
    {
        AddWindow(1000, 600, false);
        SetFPS(60);        
        width = clientWidth/scale;
        height = clientHeight/scale;
        std::srand(0);
        InitDots();
        return true;
    }
    myColor& Pixel(vec2 pos) { return GetLinePointer((int)(pos.y()))[(int)pos.x()]; }

    float rnd() { return ((float) std::rand()) / RAND_MAX; }
    float dist2(vec2 v1, vec2 v2) { vec2 dv = v2 - v1; return dv.x() * dv.x() + dv.y() * dv.y(); }
    float dist(vec2 v1, vec2 v2) { return sqrt(dist2(v1, v2)); }
    float Distance2(vec2 v1, vec2 v2) { vec2 dv = v2 - v1; return dv.x() * dv.x() + dv.y() * dv.y(); }
    float attraction(vec2 v1, vec2 v2)
    {
        vec2 dv = v2 - v1; 
        float d2 =  dv.x() * dv.x() + dv.y() * dv.y();
        float x = sqrt(d2);
        
        const float a = 5;
        const float c = 1;
        const float d = 2;
        return 2*exp(-(x - a) * (x - a) / c) - 50 * exp(-(x - a/2) * (x - a/2) / d);
        
    }
    void fade()
    {        
        ForEachPixel([this](int x, int y, myColor& c) {const int fader = 100;
            c = myColor(fader * c.r / 255, fader * c.g / 255, fader * c.b / 255);
            });
    }
    void OnDraw() override
    {
        fade();
        dots[0].pos.x() = (float)mouseX/scale;
        dots[0].pos.y() = (float)mouseY/scale;
        
    }
        
    void DrawDot(int nDot)
    {
        auto &d = dots[nDot];
        //GetLinePointer((int)d.pos.y())[(int)d.pos.x()] = d.color;
        FillCircle((scale*d.pos).cast<int>(), scale, d.color);
    }
    void DrawDots()
    {
#pragma omp parallel for
        for (int i = 0; i < (int)dots.size(); i++)
        {
            DrawDot(i);
            if (i != 0)
                MoveDot(i);
        }
    }
    bool OnUpdate() override
    {   
        DrawDots();
        return true;

        //return false;
    }
};
