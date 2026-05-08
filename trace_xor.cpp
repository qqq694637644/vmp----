#include <Windows.h>
#include <DbgHelp.h>

#include <array>
#include <cwchar>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#pragma comment(lib, "Dbghelp.lib")

namespace
{
constexpr DWORD kTrapFlag = 0x100;
constexpr BYTE kInt3Opcode = 0xCC;

struct Breakpoint
{
    DWORD64 address = 0;
    BYTE originalByte = 0;
    DWORD oldProtect = 0;
    bool armed = false;
};

struct TraceState
{
    HANDLE process = nullptr;
    HANDLE thread = nullptr;
    DWORD64 moduleBase = 0;
    DWORD64 functionRva = 0;
    DWORD64 functionAddress = 0;
    DWORD64 functionSize = 0;
    DWORD64 returnAddress = 0;
    std::uint32_t plaintextValue = 0;
    std::uint32_t keyValue = 0;
    DWORD64 resultValue = 0;
    Breakpoint entryBreakpoint;
    bool tracing = false;
    std::uint64_t stepIndex = 0;
};

std::array<BYTE, 4> DwordToBytes(std::uint32_t value)
{
    return {
        static_cast<BYTE>(value & 0xFF),
        static_cast<BYTE>((value >> 8) & 0xFF),
        static_cast<BYTE>((value >> 16) & 0xFF),
        static_cast<BYTE>((value >> 24) & 0xFF),
    };
}

std::string ToHex64(DWORD64 value)
{
    std::ostringstream stream;
    stream << "0x" << std::uppercase << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::string BytesToHex(const BYTE *bytes, SIZE_T count)
{
    std::ostringstream stream;
    stream << std::uppercase << std::hex << std::setfill('0');
    for (SIZE_T index = 0; index < count; ++index)
    {
        if (index != 0)
        {
            stream << ' ';
        }
        stream << std::setw(2) << static_cast<unsigned int>(bytes[index]);
    }
    return stream.str();
}

void PrintLine(const std::string &text)
{
    std::cout << text << std::endl;
}

[[noreturn]] void Fail(const std::string &message)
{
    PrintLine(std::string(u8"错误：") + message + u8"，GetLastError=" + std::to_string(GetLastError()));
    ExitProcess(1);
}

std::wstring GetParentDirectory(const std::wstring &path)
{
    const std::size_t position = path.find_last_of(L"\\/");
    if (position == std::wstring::npos)
    {
        return L".";
    }
    return path.substr(0, position);
}

std::wstring QuoteArgument(const std::wstring &argument)
{
    if (argument.empty())
    {
        return L"\"\"";
    }

    std::wstring quoted = L"\"";
    std::size_t backslashCount = 0;
    for (wchar_t character : argument)
    {
        if (character == L'\\')
        {
            ++backslashCount;
            continue;
        }

        if (character == L'"')
        {
            quoted.append(backslashCount * 2 + 1, L'\\');
            quoted.push_back(L'"');
            backslashCount = 0;
            continue;
        }

        quoted.append(backslashCount, L'\\');
        backslashCount = 0;
        quoted.push_back(character);
    }

    quoted.append(backslashCount * 2, L'\\');
    quoted.push_back(L'"');
    return quoted;
}

bool ReadProcessBytes(HANDLE process, DWORD64 address, BYTE *buffer, SIZE_T length, SIZE_T &readCount)
{
    readCount = 0;
    return ReadProcessMemory(process, reinterpret_cast<LPCVOID>(address), buffer, length, &readCount) != 0;
}

bool SetSoftwareBreakpoint(HANDLE process, Breakpoint &breakpoint)
{
    DWORD oldProtect = 0;
    if (VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, PAGE_EXECUTE_READWRITE, &oldProtect) == 0)
    {
        return false;
    }

    SIZE_T readCount = 0;
    if (ReadProcessBytes(process, breakpoint.address, &breakpoint.originalByte, 1, readCount) == false || readCount != 1)
    {
        DWORD ignored = 0;
        VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, oldProtect, &ignored);
        return false;
    }

    const BYTE int3 = kInt3Opcode;
    SIZE_T writtenCount = 0;
    if (WriteProcessMemory(process, reinterpret_cast<LPVOID>(breakpoint.address), &int3, 1, &writtenCount) == 0 || writtenCount != 1)
    {
        DWORD ignored = 0;
        VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, oldProtect, &ignored);
        return false;
    }

    FlushInstructionCache(process, reinterpret_cast<LPCVOID>(breakpoint.address), 1);

    DWORD ignored = 0;
    VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, oldProtect, &ignored);

    breakpoint.oldProtect = oldProtect;
    breakpoint.armed = true;
    return true;
}

bool RestoreSoftwareBreakpoint(HANDLE process, Breakpoint &breakpoint)
{
    if (!breakpoint.armed)
    {
        return true;
    }

    DWORD oldProtect = 0;
    if (VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, PAGE_EXECUTE_READWRITE, &oldProtect) == 0)
    {
        return false;
    }

    SIZE_T writtenCount = 0;
    if (WriteProcessMemory(process, reinterpret_cast<LPVOID>(breakpoint.address), &breakpoint.originalByte, 1, &writtenCount) == 0 || writtenCount != 1)
    {
        DWORD ignored = 0;
        VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, oldProtect, &ignored);
        return false;
    }

    FlushInstructionCache(process, reinterpret_cast<LPCVOID>(breakpoint.address), 1);

    DWORD ignored = 0;
    VirtualProtectEx(process, reinterpret_cast<LPVOID>(breakpoint.address), 1, oldProtect, &ignored);
    breakpoint.armed = false;
    return true;
}

struct SymbolSearchContext
{
    DWORD64 address = 0;
    DWORD64 size = 0;
    bool found = false;
};

BOOL CALLBACK EnumSymbolCallback(PSYMBOL_INFOW symbol, ULONG, PVOID context)
{
    auto *result = static_cast<SymbolSearchContext *>(context);
    if (symbol->Name != nullptr && std::wcsstr(symbol->Name, L"XorTransform") != nullptr)
    {
        result->address = symbol->Address;
        result->size = symbol->Size;
        result->found = true;
        return FALSE;
    }
    return TRUE;
}

bool ResolveXorTransform(HANDLE process, DWORD64 moduleBase, const std::wstring &symbolPath, TraceState &state)
{
    const std::wstring searchPath = GetParentDirectory(symbolPath);
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (SymInitializeW(process, searchPath.c_str(), FALSE) == 0)
    {
        return false;
    }

    // 明确指定符号来源模块，避免依赖保护后目标文件里已经消失的 PDB/符号信息。
    const DWORD64 loadedBase = SymLoadModuleExW(process, nullptr, symbolPath.c_str(), nullptr, moduleBase, 0, nullptr, 0);
    if (loadedBase == 0)
    {
        SymCleanup(process);
        return false;
    }

    SymbolSearchContext searchContext;
    if (SymEnumSymbolsW(process, loadedBase, nullptr, EnumSymbolCallback, &searchContext) == 0 || searchContext.found == false)
    {
        SymCleanup(process);
        return false;
    }

    state.moduleBase = loadedBase;
    state.functionRva = searchContext.address - loadedBase;
    state.functionAddress = searchContext.address;
    state.functionSize = searchContext.size;
    return true;
}

void LogInstruction(TraceState &state, DWORD64 address)
{
    BYTE bytes[16] = {};
    SIZE_T readCount = 0;
    if (ReadProcessBytes(state.process, address, bytes, sizeof(bytes), readCount) == false || readCount == 0)
    {
        Fail("读取指令字节失败");
    }

    std::ostringstream stream;
    stream << u8"步骤 " << std::setw(6) << std::setfill('0') << state.stepIndex
           << u8" | RIP=" << ToHex64(address)
           << u8" | 字节=" << BytesToHex(bytes, readCount);

    IMAGEHLP_LINE64 lineInfo{};
    lineInfo.SizeOfStruct = sizeof(lineInfo);
    DWORD displacement = 0;
    if (SymGetLineFromAddr64(state.process, address, &displacement, &lineInfo) != 0)
    {
        stream << u8" | 行号=" << lineInfo.LineNumber;
    }

    PrintLine(stream.str());
}

void StartTrace(TraceState &state)
{
    CONTEXT context{};
    context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (GetThreadContext(state.thread, &context) == 0)
    {
        Fail("获取线程上下文失败");
    }

    state.returnAddress = 0;
    SIZE_T readCount = 0;
    if (ReadProcessBytes(state.process, context.Rsp, reinterpret_cast<BYTE *>(&state.returnAddress), sizeof(state.returnAddress), readCount) == false || readCount != sizeof(state.returnAddress))
    {
        Fail("读取返回地址失败");
    }

    state.plaintextValue = static_cast<std::uint32_t>(context.Rcx);
    state.keyValue = static_cast<std::uint32_t>(context.Rdx);

    if (RestoreSoftwareBreakpoint(state.process, state.entryBreakpoint) == false)
    {
        Fail("恢复入口断点失败");
    }

    context.Rip = state.entryBreakpoint.address;
    context.EFlags |= kTrapFlag;
    if (SetThreadContext(state.thread, &context) == 0)
    {
        Fail("设置线程上下文失败");
    }

    state.tracing = true;
    state.stepIndex = 1;
    PrintLine(std::string(u8"已进入 XorTransform，返回地址=") + ToHex64(state.returnAddress));
    PrintLine(
        std::string(u8"参数：RSP=") + ToHex64(context.Rsp) +
        u8"，RCX=" + ToHex64(context.Rcx) +
        u8"，RDX=" + ToHex64(context.Rdx) +
        u8"，plaintext=" + BytesToHex(DwordToBytes(state.plaintextValue).data(), 4) +
        u8"，key=" + BytesToHex(DwordToBytes(state.keyValue).data(), 4));
    LogInstruction(state, state.entryBreakpoint.address);
}

void HandleSingleStep(TraceState &state)
{
    CONTEXT context{};
    context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
    if (GetThreadContext(state.thread, &context) == 0)
    {
        Fail("获取单步上下文失败");
    }

    if (context.Rip == state.returnAddress)
    {
        state.resultValue = context.Rax;
        const std::uint32_t result32 = static_cast<std::uint32_t>(state.resultValue);
        PrintLine(std::string(u8"返回值：RAX=") + ToHex64(state.resultValue) + u8"，bytes=" + BytesToHex(DwordToBytes(result32).data(), 4));
        state.tracing = false;
        PrintLine(std::string(u8"已离开 XorTransform，步骤数=") + std::to_string(state.stepIndex));
        return;
    }

    ++state.stepIndex;
    LogInstruction(state, context.Rip);

    context.EFlags |= kTrapFlag;
    if (SetThreadContext(state.thread, &context) == 0)
    {
        Fail("恢复单步标志失败");
    }
}
} // namespace

int wmain(int argc, wchar_t *argv[])
{
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);

    if (argc < 2)
    {
        PrintLine(u8"用法：trace_xor.exe --symbols <带符号模块> <目标程序> [参数...]");
        return 1;
    }

    std::wstring symbolPath;
    std::wstring targetPath;
    std::vector<std::wstring> targetArgs;

    for (int index = 1; index < argc; ++index)
    {
        const std::wstring argument = argv[index];
        if (argument == L"--symbols")
        {
            if (index + 1 >= argc)
            {
                PrintLine(u8"错误：--symbols 后面必须跟一个带符号模块路径");
                return 1;
            }

            symbolPath = argv[++index];
            continue;
        }

        if (argument == L"--help" || argument == L"-h" || argument == L"/?")
        {
            PrintLine(u8"用法：trace_xor.exe --symbols <带符号模块> <目标程序> [参数...]");
            return 0;
        }

        if (targetPath.empty())
        {
            targetPath = argument;
            continue;
        }

        targetArgs.push_back(argument);
    }

    if (symbolPath.empty())
    {
        PrintLine(u8"错误：必须显式提供 --symbols <带符号模块>");
        return 1;
    }

    if (targetPath.empty())
    {
        PrintLine(u8"错误：缺少目标程序路径");
        return 1;
    }

    std::wstring commandLine = QuoteArgument(targetPath);
    for (const std::wstring &argument : targetArgs)
    {
        commandLine.push_back(L' ');
        commandLine.append(QuoteArgument(argument));
    }

    STARTUPINFOW startupInfo{};
    startupInfo.cb = sizeof(startupInfo);

    PROCESS_INFORMATION processInfo{};
    if (CreateProcessW(
            nullptr,
            commandLine.data(),
            nullptr,
            nullptr,
            FALSE,
            DEBUG_ONLY_THIS_PROCESS | CREATE_NO_WINDOW,
            nullptr,
            nullptr,
            &startupInfo,
            &processInfo) == 0)
    {
        Fail("启动目标进程失败");
    }

    TraceState state;
    state.process = processInfo.hProcess;
    state.thread = processInfo.hThread;

    bool running = true;

    while (running)
    {
        DEBUG_EVENT debugEvent{};
        if (WaitForDebugEvent(&debugEvent, INFINITE) == 0)
        {
            Fail("等待调试事件失败");
        }

        DWORD continueStatus = DBG_CONTINUE;

        switch (debugEvent.dwDebugEventCode)
        {
        case CREATE_PROCESS_DEBUG_EVENT:
        {
            if (ResolveXorTransform(processInfo.hProcess, reinterpret_cast<DWORD64>(debugEvent.u.CreateProcessInfo.lpBaseOfImage), symbolPath, state) == false)
            {
                Fail("解析 XorTransform 符号失败");
            }

            state.entryBreakpoint.address = state.functionAddress;
            if (SetSoftwareBreakpoint(state.process, state.entryBreakpoint) == false)
            {
                Fail("设置入口断点失败");
            }

            PrintLine(std::string(u8"已定位 XorTransform，地址=") + ToHex64(state.functionAddress) + u8"，RVA=" + ToHex64(state.functionRva) + u8"，大小=" + std::to_string(state.functionSize));

            if (debugEvent.u.CreateProcessInfo.hFile != nullptr)
            {
                CloseHandle(debugEvent.u.CreateProcessInfo.hFile);
            }
            break;
        }

        case LOAD_DLL_DEBUG_EVENT:
            if (debugEvent.u.LoadDll.hFile != nullptr)
            {
                CloseHandle(debugEvent.u.LoadDll.hFile);
            }
            break;

        case EXCEPTION_DEBUG_EVENT:
        {
            const DWORD exceptionCode = debugEvent.u.Exception.ExceptionRecord.ExceptionCode;
            const DWORD64 exceptionAddress = reinterpret_cast<DWORD64>(debugEvent.u.Exception.ExceptionRecord.ExceptionAddress);

            if (exceptionCode == EXCEPTION_BREAKPOINT && state.entryBreakpoint.armed && exceptionAddress == state.entryBreakpoint.address)
            {
                StartTrace(state);
                continueStatus = DBG_CONTINUE;
                break;
            }

            if (exceptionCode == EXCEPTION_SINGLE_STEP && state.tracing)
            {
                HandleSingleStep(state);
                continueStatus = DBG_CONTINUE;
                break;
            }

            if (exceptionCode == EXCEPTION_BREAKPOINT || exceptionCode == EXCEPTION_SINGLE_STEP)
            {
                continueStatus = DBG_CONTINUE;
            }
            else
            {
                continueStatus = DBG_EXCEPTION_NOT_HANDLED;
            }
            break;
        }

        case EXIT_PROCESS_DEBUG_EVENT:
            running = false;
            break;

        default:
            break;
        }

        if (ContinueDebugEvent(debugEvent.dwProcessId, debugEvent.dwThreadId, continueStatus) == 0)
        {
            Fail("继续调试事件失败");
        }
    }

    SymCleanup(processInfo.hProcess);
    CloseHandle(processInfo.hThread);
    CloseHandle(processInfo.hProcess);

    return 0;
}
