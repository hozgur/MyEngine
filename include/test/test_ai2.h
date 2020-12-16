#pragma once
#include <unknwn.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.h>   // for string
#include <winrt/Windows.Graphics.Imaging.h> // for softwarebitmap
#include <winrt/Windows.Media.h> // for videoframe
#include <winrt/Windows.Storage.h> // for file io
#include <MemoryBuffer.h>
#include <algorithm>
#pragma comment(lib, "windowsapp")

#include "mypy.h"
#include "myxml.h"
using namespace My;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Foundation;
using namespace Windows::Foundation;

class MyEngine : public My::Engine
{
public:
    int Width = 720;
    int Height = 720;
    LearningModel model = LearningModel::LoadFromFilePath(myfs::s2w(myfs::path("asset/candy.onnx")));
    LearningModelDeviceKind deviceKind = LearningModelDeviceKind::Cpu;
    LearningModelSession session = nullptr;
    LearningModelBinding binding = nullptr;

    TensorFloat inputTensor = nullptr;
    TensorFloat outputTensor = nullptr;
    float* inputData = nullptr;
    float* outputData = nullptr;
    MyEngine(const char* path) :My::Engine(path)
    {

    }

    
    
    void ExtractDataPointers()
    {
        IMemoryBufferReference ref = inputTensor.CreateReference();
        winrt::com_ptr<IMemoryBufferByteAccess> mbba = ref.as<IMemoryBufferByteAccess>();
        {
            BYTE* data = nullptr;
            UINT32 capacity = 0;
            if (SUCCEEDED(mbba->GetBuffer(&data, &capacity)))
            {
                inputData = (float*)data;
            }
        }
        

        IMemoryBufferReference ref2 = outputTensor.CreateReference();
        winrt::com_ptr<IMemoryBufferByteAccess> mbba2 = ref2.as<IMemoryBufferByteAccess>();
        {
            BYTE* data = nullptr;
            UINT32 capacity = 0;
            if (SUCCEEDED(mbba2->GetBuffer(&data, &capacity)))
            {
                outputData = (float*)data;
            }
        }
        

    }

    void eval()
    {
        
        auto results = session.Evaluate(binding, L"Run");
        //auto resultTensor = results.Outputs().Lookup(L"outputImage").as<TensorFloat>();
        //auto resultVector = resultTensor.GetAsVectorView();
    }

    bool OnStart() override
    {
        
        session = LearningModelSession{ model, LearningModelDevice(deviceKind) };
        
        inputTensor = TensorFloat::Create({ 1,3,Width,Height });
        outputTensor = TensorFloat::Create({ 1,3,Width,Height });
        ExtractDataPointers();
        binding = LearningModelBinding{ session };
        binding.Bind(L"inputImage", inputTensor);
        binding.Bind(L"outputImage", outputTensor);

        AddWindow(720, 720);
        eval();
        draw();
        return true;
    }
    uint8_t clamp(int val)
    {
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        return val;
    }
    void draw()
    {
        int offset = mouseY * Width + mouseX;

        int size = Width * Height;
        if ((offset >= 0) && (offset < size))
        {
            *(inputData + offset) = 255;
            *(inputData + offset + size) = 0;
            *(inputData + offset + 2*size) = 255;
        }
        eval();
        debug << "eval";
#pragma omp parallel for
        for (int y = 0; y < Height; y++)
        {
            Color* p = GetLinePointer(y);
            float* red = outputData + y * Width;
            float* green = outputData + (y * Width) + size;
            float* blue = outputData + (y * Width) + 2 * size;
            for (int x = 0; x < Width; x++)
            {
                p[x] = Color(clamp(red[x]), clamp(green[x]), clamp(blue[x]));
            }
        }
    }

    void OnDraw()
    {
        draw();
        
    }
};