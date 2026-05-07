#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "VMProtectSDK.h"

namespace
{
constexpr std::size_t kLength = 4;

std::string BytesToHex(const unsigned char *data, std::size_t size)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < size; ++index)
    {
        stream << std::setw(2) << static_cast<int>(data[index]);
    }
    return stream.str();
}

__declspec(noinline) void XorTransform(const unsigned char *plaintext,
                                       const unsigned char *key,
                                       unsigned char *output,
                                       std::size_t length)
{
    // 这里保留最小可逆算法，方便后续用 Triton 直接还原出 plaintext/key 到 output 的公式。
    VMProtectBeginVirtualization("xor_transform");

    for (std::size_t index = 0; index < length; ++index)
    {
        output[index] = plaintext[index] ^ key[index];
    }

    VMProtectEnd();
}
} // namespace

int main()
{
    constexpr unsigned char plaintext[kLength] = {'1', '2', '3', '4'};
    constexpr unsigned char key[kLength] = {'k', 'e', 'y', '!'};
    std::array<unsigned char, kLength> ciphertext{};

    XorTransform(plaintext, key, ciphertext.data(), ciphertext.size());
    std::cout << "plaintext : " << BytesToHex(plaintext, kLength) << std::endl;
    std::cout << "key       : " << BytesToHex(key, kLength) << std::endl;
    std::cout << "ciphertext: " << BytesToHex(ciphertext.data(), ciphertext.size()) << std::endl;
    return 0;
}
