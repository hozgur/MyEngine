#pragma once
#include "mypy.h"
#include "myxml.h"
using namespace My;

class MyEngine : public My::Engine
{
public:

    MyEngine(const char* path) :My::Engine(path)
    {
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
        test3();
        return true;
    }
};