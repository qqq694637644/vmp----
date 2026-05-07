[CmdletBinding()]
param(
    [string]$TargetFile = "build\encrypt_demo.exe",
    [string[]]$TargetArgs = @(),
    [string]$TraceExe = "build\trace_xor.exe"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $repoRoot "build.ps1") -SourceFile "encrypt_demo.cpp" -OutputName "encrypt_demo.exe"
if ($LASTEXITCODE -ne 0) {
    throw "target build failed"
}

& (Join-Path $repoRoot "build.ps1") -SourceFile "trace_xor.cpp" -OutputName "trace_xor.exe"
if ($LASTEXITCODE -ne 0) {
    throw "tracer build failed"
}

$tracePath = Join-Path $repoRoot $TraceExe
$targetPath = Join-Path $repoRoot $TargetFile

if (-not (Test-Path $tracePath -PathType Leaf)) {
    throw "tracer not found: $tracePath"
}

if (-not (Test-Path $targetPath -PathType Leaf)) {
    throw "target not found: $targetPath"
}

$arguments = @($targetPath) + $TargetArgs
& $tracePath @arguments
