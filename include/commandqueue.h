#pragma once
#include <mutex>
enum class myCommands
{
    Navigate,
    NavigateContent,
    SetScript,
    PostWebMessage
};

typedef std::variant<long, double, std::string> engvariant;
struct myCommand
{
    myHandle id;
    myCommands commandID;
    std::vector<engvariant> params;
};

struct myCommandQueue
{
    std::mutex mtx;
    std::queue<myCommand> commands;
    void push(myCommand cmd)
    {
        std::unique_lock<std::mutex> lck(mtx, std::defer_lock);
        lck.lock();
        commands.push(cmd);
        lck.unlock();
    }
    bool pop(myCommand& cmd)
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
