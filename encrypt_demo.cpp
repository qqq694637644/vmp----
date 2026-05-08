#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "VMProtectSDK.h"

namespace
{
std::string TextToHex(std::uint32_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < sizeof(value); ++index)
    {
        stream << std::setw(2) << static_cast<unsigned int>((value >> (index * 8)) & 0xFF);
    }
    return stream.str();
}

__declspec(noinline) std::uint32_t XorTransform(std::uint32_t plaintext, std::uint32_t key)
{
    // 这里保留最小可逆算法，方便后续直接从返回值恢复出 plaintext/key 到结果的公式。
    VMProtectBeginVirtualization("xor_transform");
    const std::uint32_t result = plaintext ^ key;
    VMProtectEnd();
    return result;
}
} // namespace

int main()
{
    constexpr std::uint32_t plaintext = 0x34333231u;
    constexpr std::uint32_t key = 0x2179656bu;

    const std::uint32_t result = XorTransform(plaintext, key);
    std::cout << "plaintext : 1234" << std::endl;
    std::cout << "key       : key!" << std::endl;
    std::cout << "result    : 0x" << TextToHex(result) << std::endl;
    return 0;
}
