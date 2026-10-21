// Record analyzer arguments without routing them through a command shell.
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--version")
        {
            std::cout << "clang-tidy version 19.0.0\n";
            return 0;
        }
    }
    const char* filename = std::getenv("LINALG_CI_TOOL_LOG");
    if (!filename)
    {
        return 1;
    }
    std::ofstream log(filename, std::ios::app);
    log << '[';
    for (int i = 1; i < argc; ++i)
    {
        log << (i == 1 ? "\"" : ",\"");
        for (const unsigned char ch : std::string(argv[i]))
        {
            if (ch == '"' || ch == '\\')
            {
                log << '\\' << ch;
            }
            else if (ch < 0x20)
            {
                constexpr char hex[] = "0123456789abcdef";
                log << "\\u00" << hex[ch >> 4] << hex[ch & 0xf];
            }
            else
            {
                log << ch;
            }
        }
        log << '"';
    }
    log << "]\n";
    return log ? 0 : 1;
}
