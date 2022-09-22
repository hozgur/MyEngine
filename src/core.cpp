#include "core.h"

#ifdef WINDOWS
#include "windows/windowscore.h"
#endif // WINDOWS
namespace fs = std::filesystem;
namespace myfs
{
#ifdef WINDOWS
    std::string w2s(std::wstring_view wstring)
    {
        return ConvertW2S(wstring);
    }
    std::wstring s2w(std::string_view string)
    {
        return ConvertS2W(string);
    }
#else
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::string w2s(std::wstring wstring)
    {
        return converter.to_bytes(wstring);
    }
    std::wstring s2w(std::string string)
    {
        return converter.from_bytes(string);
    }
#endif
    
    
    bool exits(const std::string& path, const std::string& filename)
    {
        for (const auto& entry : fs::directory_iterator(path))
        {
            std::string name = entry.path().filename().string();
            if (name == filename)
                return true;
        }
        return false;
    }


    std::string root()
    {
        fs::path cpath = fs::current_path();
        //debug << isFileExists(cpath);
        while (true)
        {
            if (exits(cpath.string(), "myroot.txt"))
                return cpath.string();

            fs::path parent = cpath.parent_path();

            if (parent == cpath)
                return std::string();
            cpath = parent;
            //debug << cpath << "\n";
        }
        return std::string();
    }

    std::string path(std::string directory)
    {
        fs::path path = root();
        path /= directory;
        return path.make_preferred().string();
    }

	std::string path(std::string directory, std::string filename)
	{
		fs::path path = root();
		path /= directory;
		path /= filename;
		return path.make_preferred().string();
	}
    
    std::string getEnv(std::string env)
    {
        size_t len;
        char* pValue;
        errno_t err = _dupenv_s(&pValue, &len, env.c_str());
        if (err) return "";
		std::string value = pValue;        
        free(pValue);
        return value;
    }    
}
