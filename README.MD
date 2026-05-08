使用triton分析vmp虚拟化测试实验
用来验证triton恢复vmp虚拟化保护算法能力

build.py是encrypt_demo.cpp的编译脚本,使用msvc amd64编译,固定基址
protect.ps1是vmprotect 3.9.1的加密脚本,用来加裸虚拟化保护

trace_xor.cpp 用来做指令追踪,内存状态保存
trace_xor_build.ps1是编译trace_xor.cpp脚本