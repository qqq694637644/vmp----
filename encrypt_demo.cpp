#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "VMProtectSDK.h"

namespace
{
constexpr unsigned char kXorKey = 0x5A;

std::string BytesToHex(const std::vector<unsigned char> &data)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char value : data)
    {
        stream << std::setw(2) << static_cast<int>(value);
    }
    return stream.str();
}

__declspec(noinline) std::vector<unsigned char> XorTransform(const std::string &input)
{
    // 这里故意只保留一个最小算法示例，核心目的是演示 VMP 虚拟化如何包裹敏感逻辑。
    VMProtectBeginVirtualization("xor_transform");

    std::vector<unsigned char> output;
    output.reserve(input.size());

    for (unsigned char ch : input)
    {
        output.push_back(ch ^ kXorKey);
    }

    VMProtectEnd();
    return output;
}
} // namespace

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: encrypt_demo.exe <plaintext>" << std::endl;
        return 1;
    }

    const std::string plaintext = argv[1];
    if (plaintext.empty())
    {
        throw std::invalid_argument("plaintext must not be empty");
    }

    const std::vector<unsigned char> ciphertext = XorTransform(plaintext);
    std::cout << "plaintext : " << plaintext << std::endl;
    std::cout << "ciphertext: " << BytesToHex(ciphertext) << std::endl;
    return 0;
}
