#include <cstddef>
#include <cstdint>
#include <intrin.h>
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

constexpr std::uint32_t Rotl32(std::uint32_t value, unsigned int shift)
{
    shift &= 31U;
    return (value << shift) | (value >> ((32U - shift) & 31U));
}

__declspec(noinline) std::uint32_t XorTransform(std::uint32_t plaintext, std::uint32_t key)
{
    // 这里再加一层字节重排和常量乘法，专门压一下 VMP + Triton 的恢复链路。
    VMProtectBeginVirtualization("xor_transform");
    const std::uint32_t step1 = plaintext + 0x13579BDFu;
    const std::uint32_t step2 = Rotl32(key ^ 0x2468ACE0u, 7);
    const std::uint32_t step3 = _byteswap_ulong(step1 ^ step2);
    const std::uint32_t step4 = Rotl32(plaintext ^ 0x11223344u, 11);
    const std::uint32_t step5 = (step3 + step4) * 0x9E3779B1u;
    const std::uint32_t step6 = step5 ^ (key + 0x0F1E2D3Cu);
    const std::uint32_t result = Rotl32(step6, 3) ^ 0xA5A5A5A5u;
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
