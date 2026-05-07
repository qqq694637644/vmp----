#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_VMP_ROOT = Path(r"F:\Maye-13.6.0.230528\Tools\VMProtect3.9.4")
DEFAULT_VCVARS64 = Path(
    r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
)

def find_vcvars64(explicit: str | None) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"找不到 vcvars64.bat: {candidate}")

    for candidate in (
        DEFAULT_VCVARS64,
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio"
        / "2022"
        / "Professional"
        / "VC"
        / "Auxiliary"
        / "Build"
        / "vcvars64.bat",
    ):
        if candidate.exists():
            return candidate

    for vswhere in (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio"
        / "Installer"
        / "vswhere.exe",
    ):
        if not vswhere.exists():
            continue

        query = [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
        result = subprocess.run(query, capture_output=True, text=True, check=True)
        installation_path = result.stdout.strip().splitlines()
        if not installation_path:
            continue

        candidate = Path(installation_path[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "无法找到 vcvars64.bat。请安装 Visual Studio Build Tools，或通过 --vcvars 指定路径。"
    )


def escape_cmd_quotes(text: str) -> str:
    return text.replace('"', '""')


def build_command_line(
    vcvars64: Path,
    vmp_include_dir: Path,
    vmp_lib_dir: Path,
    source_path: Path,
    output_path: Path,
    pdb_path: Path,
) -> str:
    body = (
        f'pushd "{vcvars64.parent}" && '
        "call vcvars64.bat && "
        "popd && "
        "cl /nologo /std:c++17 /utf-8 /Od /Ob0 /EHsc /Zi /FS "
        f'/I"{vmp_include_dir}" '
        f'/Fd:"{pdb_path}" "{source_path}" '
        f'/link /OUT:"{output_path}" /LIBPATH:"{vmp_lib_dir}" /DYNAMICBASE:NO'
    )
    return f'cmd.exe /d /s /c "{escape_cmd_quotes(body)}"'


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local VMP demo with MSVC.")
    parser.add_argument("--source-file", default="encrypt_demo.cpp")
    parser.add_argument("--output-dir", default="build")
    parser.add_argument("--output-name", default="encrypt_demo.exe")
    parser.add_argument("--vcvars", help="Path to vcvars64.bat")
    parser.add_argument("--vmp-root", default=str(DEFAULT_VMP_ROOT))
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it")
    args = parser.parse_args()

    source_path = (ROOT / args.source_file).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    output_path = (output_dir / args.output_name).resolve()
    pdb_path = (output_dir / (Path(args.output_name).stem + ".pdb")).resolve()
    vmp_root = Path(args.vmp_root).resolve()
    vmp_include_dir = vmp_root / "Include" / "C"
    vmp_lib_dir = vmp_root / "Lib" / "Windows"
    vmp_dll_path = vmp_lib_dir / "VMProtectSDK64.dll"
    vcvars64 = find_vcvars64(args.vcvars)

    if not source_path.exists():
        raise FileNotFoundError(f"源文件不存在: {source_path}")
    if not vmp_include_dir.exists():
        raise FileNotFoundError(f"未找到 VMP 头文件目录: {vmp_include_dir}")
    if not vmp_lib_dir.exists():
        raise FileNotFoundError(f"未找到 VMP 库目录: {vmp_lib_dir}")
    if not vmp_dll_path.exists():
        raise FileNotFoundError(f"未找到 VMP 运行时 DLL: {vmp_dll_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    command_line = build_command_line(vcvars64, vmp_include_dir, vmp_lib_dir, source_path, output_path, pdb_path)

    print(f"源文件: {source_path.relative_to(ROOT)}")
    print(f"输出目录: {output_dir.relative_to(ROOT)}")
    print(f"运行时: {vmp_dll_path}")
    print(f"> {command_line}")

    if args.dry_run:
        print(f"PDB: {pdb_path.relative_to(ROOT)}")
        print(f"产物: {output_path.relative_to(ROOT)}")
        return 0

    subprocess.run(command_line, cwd=str(ROOT), check=True)

    if not output_path.exists():
        raise FileNotFoundError(f"编译完成但未找到目标文件: {output_path}")

    shutil.copy2(vmp_dll_path, output_dir / "VMProtectSDK64.dll")

    print(f"PDB: {pdb_path.relative_to(ROOT)}")
    print(f"产物: {output_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
