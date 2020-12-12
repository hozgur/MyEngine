#pragma once
#include "my.h"
using namespace My;
#include <3rdParty/json.hpp>

using json = nlohmann::json;

class MyEngine : public My::Engine
{
public:
    

    MyEngine(const char* path) :My::Engine(path)
    {
        json j;

        // add a number that is stored as double (note the implicit conversion of j to an object)
        j["pi"] = 3.141;

        double pi = j["pi"];
        debug << "pi = " << pi << "\n";
        // add a Boolean that is stored as bool
        j["happy"] = true;

        // add a string that is stored as std::string
        j["name"] = "Niels";

        // add another null object by passing nullptr
        j["nothing"] = nullptr;

        // add an object inside the object
        j["answer"]["everything"] = 42;

        // add an array that is stored as std::vector (using an initializer list)
        j["list"] = { 1, 0, 2 };

        // add another object (using an initializer list of pairs)
        j["object"] = { {"currency", "USD"}, {"value", 42.99} };

        // instead, you could also write (which looks very similar to the JSON above)
        json j2 = {
          {"pi", 3.141},
          {"happy", true},
          {"name", "Niels"},
          {"nothing", nullptr},
          {"answer", {
            {"everything", 42}
          }},
          {"list", {1, 0, 2}},
          {"object", {
            {"currency", "USD"},
            {"value", 42.99}
          }}
        };

        debug << j2.dump() << "\n";

        // create object from string literal
        json j3 = "{ \"happy\": true, \"pi\": 3.141 }"_json;

        debug << "j3 = " << j3 << "\n";
        // or even nicer with a raw string literal
        auto j4 = R"(
                {
                    "happy": true,
                    "pi": 3.141
                }
                )"_json;

        debug << " happy = " << j4["happy"] << "\n";
    }

    bool OnStart() override
    {
        return true;
    }    
};
