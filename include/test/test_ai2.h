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
    const int inWidth = 720;
    const int inHeight = 720;
    const int outWidth = 720;
    const int outHeight = 720;
    LearningModel model = nullptr;
    LearningModelDeviceKind deviceKind = LearningModelDeviceKind::Cpu;
    LearningModelSession session = nullptr;
    LearningModelBinding binding = nullptr;

    TensorFloat inputTensor = nullptr;
    TensorFloat outputTensor = nullptr;
    float* inputData = nullptr;
    float* outputData = nullptr;
    byte* inputBuffer = nullptr;
    std::string inputName = "inputImage";
    std::string outputName = "outputImage";
    MyEngine(const char* path) :My::Engine(path)
    {
        handle h = loadImage(myfs::path("asset/FRUIT720.png"));

        object* o = Get(h);
        image<Color>* img = dynamic_cast<image<Color>*>(o);
        if (img != nullptr)
        {
            debug << " Width = " << img->getWidth();
        }
        inputBuffer = new byte[3 * inWidth * inHeight];
    }
    ~MyEngine()
    {
        delete inputBuffer;
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
        model = LearningModel::LoadFromFilePath(myfs::s2w(myfs::path("asset/candy.onnx")));
        dumpFeatures(model);
        session = LearningModelSession{ model, LearningModelDevice(deviceKind) };        
        inputTensor = TensorFloat::Create({ 1,3,inWidth,inHeight });
        outputTensor = TensorFloat::Create({ 1,3,outWidth,outHeight });
        ExtractDataPointers();                
        binding = LearningModelBinding{ session };
        binding.Bind(myfs::s2w(inputName), inputTensor);
        binding.Bind(myfs::s2w(outputName), outputTensor);

        AddWindow(720, 720);        
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
        const int offset = (int)(mouseY * clientWidth + mouseX);

        float* inp = inputData;
        const int inSize = inWidth * inHeight;
        const int outSize = outWidth * outHeight;
        if ((offset >= 0) && (offset < inSize))
        {
            *(inp + offset) = 255;
            *(inp + offset + inSize) = 0;
            *(inp + offset + 2*inSize) = 255;
        }
        
        eval();
        float* data = outputData;
        debug << "eval";
#pragma omp parallel for
        for (int y = 0; y < outHeight; y++)
        {
            Color* p = GetLinePointer(y);
            float* red = data + y * outWidth;
            float* green = data + (y * outWidth) + outSize;
            float* blue = data + (y * outWidth) + 2 * outSize;
            for (int x = 0; x < outWidth; x++)
            {
                p[x] = Color(clamp((int)red[x]), clamp((int)green[x]), clamp((int)blue[x]));
            }
        }
//#pragma omp parallel for
//        for (int a = 0; a < size * 3; a++)
//            inputData[a] = outputData[a];
    }

    void OnDraw()
    {
        draw();
        
    }

    void dumpFeatures(LearningModel model)
    {
        for (auto feat : model.InputFeatures())
        {
            if (feat.Kind() == LearningModelFeatureKind::Image)
            {
                ImageFeatureDescriptor imgDesc = feat.as<ImageFeatureDescriptor>();
                debug << "Width = " << imgDesc.Width() << " Height = " << imgDesc.Height() << " Channels = " << (int)imgDesc.BitmapPixelFormat();
            }
            else
            {
                TensorFeatureDescriptor tDesc = feat.as<TensorFeatureDescriptor>();
                debug << "Shape : ( ";
                for (auto a : tDesc.Shape())
                    debug << a << ",";
                debug << " ) ";
            }
            debug << "Name = " << myfs::w2s(feat.Name());
        }

        for (auto feat : model.OutputFeatures())
        {
            if (feat.Kind() == LearningModelFeatureKind::Image)
            {
                ImageFeatureDescriptor imgDesc = feat.as<ImageFeatureDescriptor>();
                debug << "Width = " << imgDesc.Width() << " Height = " << imgDesc.Height() << " Channels = " << (int)imgDesc.BitmapPixelFormat();
            }
            else
            {
                TensorFeatureDescriptor tDesc = feat.as<TensorFeatureDescriptor>();
                debug << "Shape : ( ";
                for (auto a : tDesc.Shape())
                    debug << a << ",";
                debug << " ) ";
            }
            debug << "Name = " << myfs::w2s(feat.Name());
        }
    }
};