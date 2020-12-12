#pragma once
#define PUGIXML_HEADER_ONLY
#define PUGIXML_NO_XPATH
#include "3rdparty/pugixml.hpp"
namespace myxml = pugi;		//this is for encapsulation of 3rd party library not for hide. In future we may want to change library with other. at this situation we add extra code here to keep main code same.