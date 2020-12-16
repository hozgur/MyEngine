#pragma once
#include <mutex>
namespace My
{
    enum class Commands
    {
        Navigate,
        NavigateContent,
        SetScript,
        PostWebMessage
    };

    typedef std::variant<long, double, std::string> engvariant;
    struct command
    {
        handle id;
        Commands commandID;
        std::vector<engvariant> params;
    };

    struct CommandQueue
    {
        std::mutex mtx;
        std::queue<command> commands;
        void push(command cmd)
        {
            std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
            lck.lock();
            commands.push(cmd);
            lck.unlock();
        }
        bool pop(command& cmd)
        {
            std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
            lck.lock();
            if (commands.size() > 0)
            {
                cmd = commands.front();
                commands.pop();
                lck.unlock();
                return true;
            }
            return false;
        }
    };
}