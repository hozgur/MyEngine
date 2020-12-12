#pragma once
// Þimdilik Ýptal
/*
//Sample Radius = 3 -> bufferHeight = 1 -> Start Process on Line = bufferHeight
//
//0***********************************
//1***********************************
//2***********************************
//3***********************************
//4***********************************
//
//Process until line = Height -1 (last line)
//
//Last Process:
//Process 


template<typename T = RIPProxy>
class LineProcessor
{
public:
    T proxy;
    int Width;
    int Height;
    short* lineBuffer;    
    LineProcessor(int width, int height,int radius) :proxy(width, height), Width(width), Height(height)
    {
     
        lineBuffer = new short[width*height];
    }
    ~LineProcessor()
    {
        delete lineBuffer;
    }
    void Process(int line)
    {
        LP.ReadLine()
    }
};
*/

#include "my.h"
#include "mymath.h"
#include "windows\windowscore.h"
#include "windows\windowsplatform.h"
//#include <../eigen-3.3.7/unsupported/Eigen/CXX11/Tensor>
using namespace My;

#include <filesystem>
namespace fs = std::filesystem;
#define MAXVALUE 4095

class RIPProxy
{
public:
    int GetWidth() { return 0; };
    int GetHeight() { return 0; };
    void ReadLine(int nLine, short* data) {};
    void WriteLine(int nLine, short* data) {};
    short Get(int x, int y) { return 0; };
    void Set(int x, int y, short val) {};
};

short RGB2Gray(const Color& c)
{
    return MAXVALUE * (30*c.r + 59*c.g + 11*c.b) / (100*255);
}
Color Gray2RGB(short val)
{
    byte g = (byte)(val >> 4);
    return Color(g, g, g);
}

class BasicRIPProxy : public RIPProxy
{
public:
    Surface* surface;
    BasicRIPProxy():surface(nullptr)
    {
    
    }
    ~BasicRIPProxy()
    {

    }
    int GetWidth(){ if (surface == nullptr) return 0; return surface->GetWidth(); }
    int GetHeight(){ if (surface == nullptr) return 0; return surface->GetHeight(); }
    virtual bool Open(std::string filePath)
    {                
        surface = LoadSurface(myfs::s2w(filePath));
        if (surface == nullptr)
        {
            debug << "error on load!";
            return false;
        }
        int w = surface->GetWidth();
        int h = surface->GetHeight();
        int d = surface->GetDepth();
        debug << "Width = " << w << " Height = " << h << " Depth = " << d << "\n";
        
        return true;
    }
    // Inherited via RIPProxy
    void ReadLine(int nLine, short* data) 
    {
        Color *p = (Color*)surface->GetLinePointer(nLine);

        for (int x = 0; x < GetWidth(); x++)
            data[x] = RGB2Gray(p[x]);
    }

    void WriteLine(int nLine, short* data) 
    {
        Color* p = (Color*)surface->GetLinePointer(nLine);

        for (int x = 0; x < GetWidth(); x++)
        {
            byte g = (byte)(data[x] >> 4);
            p[x] = Color(g, g, g);
        }            
    }

    short Get(int x, int y)
    {
        int xx = abs(x % GetWidth());
        int yy = abs(y % GetHeight());
        return RGB2Gray(((Color*)surface->GetLinePointer(yy))[xx]);
    }

    void Set(int x, int y, short val)
    {
        int xx = abs(x % GetWidth());
        int yy = abs(y % GetHeight());
        byte g = 255-(byte)(val >> 4);        
        ((Color*)surface->GetLinePointer(yy))[xx] = Color(g, g, g);
    }
};

#define MAX_THREAD 8
class PaddingProxy : public BasicRIPProxy
{
public:
    int nPadding;    
    
    PaddingProxy(int padding = 3):BasicRIPProxy(),nPadding(padding)
    {        
    }
    // Inherited via BasicRIPProxy

    int GetWidth() { if (BasicRIPProxy::GetWidth() == 0) return 0; return BasicRIPProxy::GetWidth()+ 2 * nPadding; }
    int GetHeight() { if (BasicRIPProxy::GetHeight() == 0) return 0; return BasicRIPProxy::GetHeight() + 2 * nPadding; }    

    virtual bool Open(std::string filePath)
    {
        if (BasicRIPProxy::Open(filePath) == false)
            return false;
                    
        return true;
    }

    void ReadLine(int nLine, short* data)
    {        
        if (nPadding > 0)
        {
            std::vector<short> buffer;
            buffer.reserve(GetWidth());
            int line = (nLine + 2*BasicRIPProxy::GetHeight() - nPadding) % BasicRIPProxy::GetHeight();
            BasicRIPProxy::ReadLine(line, buffer.data()+nPadding);
            memcpy(buffer.data(), buffer.data() + 1*(GetWidth() - 2 * nPadding), 2 * nPadding);
            memcpy(buffer.data() + 1*(GetWidth() - nPadding), buffer.data() + nPadding, 2 * nPadding);
            memcpy(data, buffer.data(), GetWidth()*2);
        }
        else
        {
            BasicRIPProxy::ReadLine(nLine, data);
        }
        
    }

    virtual void WriteLine(int nLine, short* data)
    {
        if (nPadding > 0)
        {
            
        }
        else
        {
            BasicRIPProxy::WriteLine(nLine, data);
        }
    }

};

class ConvProxy : public BasicRIPProxy
{
public:
    std::vector<short> kernel1d;
    short Get(int x, int y)
    {
        int c = (int) sqrt(kernel1d.size());
        int val = BasicRIPProxy::Get(x, y);
        if ((c > 0) && (val < MAXVALUE))
        {                                 
            for (int yy = -c / 2; yy <= c / 2; yy++)
            {
                for (int xx = -c / 2; xx <= c / 2; xx++)
                {                    
                    if (BasicRIPProxy::Get(x + xx, y + yy) == MAXVALUE)
                        val += kernel1d[(yy+c/2)*c+ xx + c/2];
                }                
            }
            if (val > MAXVALUE) val = MAXVALUE;
        }
        return (short) val;
    }
};

class MyEngine : public Engine
{
public:
    
    ConvProxy proxy;
    
    int scrollX = 50;
    MyEngine() :Engine()
    {
    
    }
    ~MyEngine()
    {
    }




    bool OnStart() override
    {



        SetScript(myfs::path("user/ilay.lua"));
        AddWindow(512,256,3,3);
        
        
        fs::path filename = myfs::root();
        filename /= "asset/256border.png";
        if (proxy.Open(filename.string()) == false)
            return false;

        //double k[] = { 0.1, 0.2, 1, 0.2, 0.1 };
        std::vector<double> k;
        int radius = lua.getglobal<int>("radius");
        for (int a = -radius; a <= radius; a++)
        {            
            k.push_back(My::Math::Gauss2((double)a/radius));
        }
        
        int size = k.size();
        int i = 0;
        for (double ky : k) 
            for (double kx : k)
                proxy.kernel1d.push_back((short)(ky * kx * MAXVALUE));
        
        int total = 0;
        for (uint32_t i = 0; i < proxy.kernel1d.size(); i++)
        {            
            debug << proxy.kernel1d[i] << " ";
            total += proxy.kernel1d[i];
            if (i % size == size-1) debug << "\n";
        }
        debug << "Total = " << total << "\n";
        return true;
    }

    void Draw()
    {
        //int mx = (((int)mouseX));
        //int my = (((int)mouseY));        
//#pragma omp parallel for
        for (int y = 0; y < proxy.GetHeight(); y++)
        {                        
            for (int x = 0; x < proxy.GetWidth(); x++)
                Pixel(x, y) = Gray2RGB(proxy.Get(x, y));
        }
        BasicRIPProxy* rip = (BasicRIPProxy*)&proxy;
        for (int y = 0; y < proxy.GetHeight(); y++)
        {
            for (int x = 0; x < proxy.GetWidth(); x++)
                Pixel(x + 256, y ) = Gray2RGB(rip->Get(x, y));
        }
    }

    void OnDraw() override
    {
        /*StopWatch s;
        s.Start();
        while (GetScanLine() != 500);
        s.Stop();
        SetWindowTitle(std::to_string(s.GetDuration()));*/
        Clear(Color::Black);
        Draw();
        //scrollX -= 15;
        //if (scrollX < 0) scrollX += clientWidth;
        
    }
};
