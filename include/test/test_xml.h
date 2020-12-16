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
    void test()
    {
        LearningModel model = LearningModel::LoadFromFilePath(myfs::s2w(myfs::path("asset/candy.onnx")));
        LearningModelDeviceKind deviceKind = LearningModelDeviceKind::Default;
        LearningModelSession session = nullptr;
        LearningModelBinding binding = nullptr;

        VideoFrame inputImage = nullptr;
        std::string inputPath = myfs::path("") + "asset\\FRUIT400.png";
        winrt::hstring inPath(myfs::s2w(inputPath));
        try {
            StorageFile file = StorageFile::GetFileFromPathAsync(inPath).get();
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
        int csize2 = csize *2;
        
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
    void test1()
    {
        myxml::xml_document doc;
        std::string path = myfs::path("user/tree.xml");
        //myxml::xml_parse_result result = doc.load_file(path.c_str(), myxml::parse_default, myxml::encoding_utf8);
        myxml::xml_parse_result result = doc.load_file(path.c_str());

        debug << "Load result: " << result.description() << ", mesh name: " << doc.child("mesh").attribute("name").value() << std::endl;

        if (!result)
        {
            std::cout << "XML [" << path << "] parsed with errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n";
            std::cout << "Error description: " << result.description() << "\n";
            std::cout << "Error offset: " << result.offset << " (error at [..." << (path.c_str() + result.offset) << "]\n\n";
        }
        for (myxml::xml_node tool : doc.child("mesh"))
        {
            std::string val = tool.value();
            debug << "Tool:" << val << "\n Atributes : ";

            for (myxml::xml_attribute attr : tool.attributes())
            {
                debug << " " << attr.name() << "=" << attr.value();
            }

            for (myxml::xml_node child : tool.children())
            {
                debug << ", child " << child.name();
            }

            debug << std::endl;
        }
    }

    void dump_node(myxml::xml_node node,int intent)
    {
        debug << std::string(intent, ' ') << node.name() << " :" << node.value();
        if(node.attributes_begin() != node.attributes_end())
            for (myxml::xml_attribute attr : node.attributes())
            {
                debug << " " << attr.name() << "=" << attr.value();
            }
        debug << "\n";
        if (node.first_child().value() != nullptr)
        {
            for (myxml::xml_node node : node.children())
            {
                dump_node(node, intent+3);
            }
        }
    }

    void test2()
    {
        myxml::xml_document doc;
        std::string path = myfs::path("user/webview/precompiled/vue_test_xml.html");
        std::ifstream ifs(path);
        std::string content((std::istreambuf_iterator<char>(ifs)),
            (std::istreambuf_iterator<char>()));

        //myxml::xml_parse_result result = doc.load_file(path.c_str(), myxml::parse_default, myxml::encoding_utf8);
        myxml::xml_parse_result result = doc.load_string(content.c_str());
        if (!result)
        {
            std::cout << "XML [" << path << "] parsed with errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n";
            std::cout << "Error description: " << result.description() << "\n";
            std::cout << "Error offset: " << result.offset << " (error at [..." << (content.c_str() + result.offset) << "]\n\n";
        }
        debug << "Load result: " << result.description() << ", mesh name: " << doc.child("mesh").attribute("name").value() << std::endl;

        for (myxml::xml_node node : doc.children())
        {
            dump_node(node,0);
        }
    }

    void test3()
    {
        myxml::xml_document doc;
        std::string path = myfs::path("user/webview/precompiled/xml_out.xml");

        myxml::xml_node node = doc.append_child();
        node.set_name("Hello");
        node.set_value("Deneme Value");
        myxml::xml_node node2 = node.append_child();
        node2.set_name("Hello2");
        node2.set_value("deneme2 value");
        myxml::xml_attribute attr = node2.append_attribute("class");
        attr.set_value("attr2");
        
        doc.save_file(path.c_str());

    }
    bool OnStart() override
    {
        test();
        return true;
    }
};