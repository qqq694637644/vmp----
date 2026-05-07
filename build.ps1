[CmdletBinding()]
param(
    [string]$SourceFile = "encrypt_demo.cpp",
    [string]$OutputDir = "build",
    [string]$OutputName = "encrypt_demo.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $repoRoot "build.py"

if (-not (Test-Path $pythonScript -PathType Leaf)) {
    throw "找不到 Python 编译脚本: $pythonScript"
}

& python $pythonScript --source-file $SourceFile --output-dir $OutputDir --output-name $OutputName
exit $LASTEXITCODE
