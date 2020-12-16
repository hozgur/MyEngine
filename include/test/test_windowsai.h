#pragma once
#include <winrt/Windows.AI.MachineLearning.h>
#include <winrt/Windows.Foundation.h>   // for string
#include <winrt/Windows.Graphics.Imaging.h> // for softwarebitmap
#include <winrt/Windows.Media.h> // for videoframe
#include <winrt/Windows.Storage.h> // for file io
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
class MyEngine : public My::Engine
{
public:

    MyEngine(const char* path) :My::Engine(path)
    {
    }

    void dumpFeatures(LearningModel model)
    {
        for(auto inputF :model.InputFeatures())
        {                                    
            if (inputF.Kind() == LearningModelFeatureKind::Image)
            {
                ImageFeatureDescriptor imgDesc = inputF.as<ImageFeatureDescriptor>();
                debug << "Width = " << imgDesc.Width() << " Height = " << imgDesc.Height() << " Channels = " << (int) imgDesc.BitmapPixelFormat();
            }
            else
            {
                TensorFeatureDescriptor tDesc = inputF.as<TensorFeatureDescriptor>();
                debug << "Shape : ( ";
                for (auto a : tDesc.Shape())
                    debug << a;
                debug << " ) ";
            }
            debug << "Name = " << myfs::w2s(inputF.Name());                        
        }
        /*foreach(var outputF in m_model.OutputFeatures)
        {
            Debug.WriteLine($"output | kind:{outputF.Kind}, name:{outputF.Name}, type:{outputF.GetType()}");
            int i = 0;
            ImageFeatureDescriptor imgDesc = outputF as ImageFeatureDescriptor;
            TensorFeatureDescriptor tfDesc = outputF as TensorFeatureDescriptor;
            m_outWidth = (uint)(imgDesc == null ? tfDesc.Shape[3] : imgDesc.Width);
            m_outHeight = (uint)(imgDesc == null ? tfDesc.Shape[2] : imgDesc.Height);
            m_outName = outputF.Name;

            Debug.WriteLine($"N: {(imgDesc == null ? tfDesc.Shape[0] : 1)}, " +
                $"Channel: {(imgDesc == null ? tfDesc.Shape[1].ToString() : imgDesc.BitmapPixelFormat.ToString())}, " +
                $"Height:{(imgDesc == null ? tfDesc.Shape[2] : imgDesc.Height)}, " +
                $"Width: {(imgDesc == null ? tfDesc.Shape[3] : imgDesc.Width)}");
        }*/
    }

    void test()
    {
        LearningModel model = LearningModel::LoadFromFilePath(myfs::s2w(myfs::path("asset/candy.onnx")));
        dumpFeatures(model);
        return;
        LearningModelDeviceKind deviceKind = LearningModelDeviceKind::Default;
        LearningModelSession session = nullptr;
        LearningModelBinding binding = nullptr;

        VideoFrame inputImage = nullptr;
        std::string inputPath = myfs::path("") + "asset\\FRUIT400.png";        
        try {
            StorageFile file = StorageFile::GetFileFromPathAsync(myfs::s2w(inputPath)).get();
            auto stream = file.OpenAsync(FileAccessMode::Read).get();
            BitmapDecoder decoder = BitmapDecoder::CreateAsync(stream).get();
            SoftwareBitmap softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
            inputImage = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
        }
        catch (...) {
            debug << "Error on Image Load \n";
            return;
        }
        session = LearningModelSession{ model, LearningModelDevice(deviceKind) };
        
        binding = LearningModelBinding{ session };
        // bind the intput image
        binding.Bind(L"input", ImageFeatureValue::CreateFromVideoFrame(inputImage));
        // bind the output
        std::vector<int64_t> shape({ 3,370,370 });
        binding.Bind(L"output", TensorFloat::Create(shape));

        auto results = session.Evaluate(binding, L"RunId");
        auto resultTensor = results.Outputs().Lookup(L"output").as<TensorFloat>();
        auto resultVector = resultTensor.GetAsVectorView();
        std::ofstream fout("data.dat", std::ios::out | std::ios::binary);
        //std::ofstream fout2("data.csv", std::ios::out);
        //std::locale cpploc{ "" };
        //fout2.imbue(cpploc);
        int i = 0;
        int csize = 370 * 370;
        int csize2 = csize * 2;

        std::vector<int> red;
        std::vector<int> green;
        std::vector<int> blue;
        for (float a : resultVector) {

            if (i < csize)
            {
                red.push_back((int)a);
            }
            else
                if (i < csize2)
                {
                    green.push_back((int)a);
                }
                else
                    blue.push_back((int)a);
            i++;
        }
        int redmax = *std::max_element(std::begin(red), std::end(red));
        int greenmax = *std::max_element(std::begin(green), std::end(green));
        int bluemax = *std::max_element(std::begin(blue), std::end(blue));

        for (int a : red)
        {
            char val = a * 255 / redmax;
            fout.write((char*)&val, 1);
        }
        for (int a : green)
        {
            char val = a * 255 / greenmax;
            fout.write((char*)&val, 1);
        }
        for (int a : blue)
        {
            char val = a * 255 / bluemax;
            fout.write((char*)&val, 1);
        }

        fout.close();
        //fout2.close();
    }
    
    bool OnStart() override
    {
        test();
        return true;
    }
};