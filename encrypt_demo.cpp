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
    // 这里保留一个稍微复杂但仍可重放的 32 位混合算法，用来验证 VMP + Triton 的公式恢复链路。
    VMProtectBeginVirtualization("xor_transform");
    const std::uint32_t step1 = plaintext + 0x13579BDFu;
    const std::uint32_t step2 = ((key ^ 0x2468ACE0u) << 7) | ((key ^ 0x2468ACE0u) >> 25);
    const std::uint32_t step3 = step1 ^ step2;
    const std::uint32_t step4 = (plaintext << 11) | (plaintext >> 21);
    const std::uint32_t step5 = step3 + step4;
    const std::uint32_t step6 = step5 ^ (key + 0x0F1E2D3Cu);
    const std::uint32_t result = ((step6 << 3) | (step6 >> 29)) ^ 0xA5A5A5A5u;
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
