#pragma once
#include "my.h"
#include <fstream>
class myParser
{
public:
	static bool parse(std::string inPath, std::string outPath, std::map<std::string, std::string> dict)
	{			
		std::string line;
		std::ifstream infile(inPath);
			
		const char* intoken = "<<<";
		const char* outtoken = ">>>";
		size_t ints1 = strlen(intoken);
		size_t ints2 = strlen(outtoken);
		if (infile.is_open())
		{
			std::ofstream outfile(outPath);
			if (outfile.is_open())
			{
				while (getline(infile, line))
				{
					size_t pos = 0;
					while (true)
					{
						std::size_t found1 = line.find(intoken, pos);
						if (found1 != std::string::npos)
						{
							pos = found1 + ints1;
							std::size_t found2 = line.find(outtoken, pos);
							if (found2 != std::string::npos)
							{
								pos = found2 + ints2;
								std::string token = line.substr(found1 + ints1, found2 - found1 - ints1);
								if (dict.count(token) > 0)
								{
									line = line.substr(0, found1) + dict[token] + line.substr(found2 + ints2);
								}
							}
							else
								break;
						}
						else
							break;
					}
					outfile << line + "\n";
				}
				outfile.close();
			}
			else
			{
				debug << "Parser Error. Output file could not be created.\n";
				return false;
			}
			infile.close();
		}
		else
		{
			debug << "Parser Error. Input file could not be opened.\n";
			return false;
		}
		return true;
	}		
};
