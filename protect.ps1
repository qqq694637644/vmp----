[CmdletBinding()]
param(
    [string]$InputFile = "build\encrypt_demo.exe",
    [string]$OutputFile = "build\encrypt_demo.protected.exe",
    [string]$ProjectFile = "build\encrypt_demo.vmp"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$inputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $InputFile))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputFile))
$projectPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $ProjectFile))

$vmpRoot = "F:\Maye-13.6.0.230528\Tools\VMProtect3.9.4"
$vmpCliPath = Join-Path $vmpRoot "VMProtect_Con.exe"

if (-not (Test-Path $inputPath -PathType Leaf)) {
    throw "待保护文件不存在: $inputPath"
}

if (-not (Test-Path $vmpCliPath -PathType Leaf)) {
    throw "未找到 VMP CLI: $vmpCliPath"
}

New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $projectPath) -Force | Out-Null

$escapedInputPath = [System.Security.SecurityElement]::Escape($inputPath)
$projectXml = @"
<?xml version="1.0" encoding="UTF-8"?>
<Document>
    <Protection InputFileName="$escapedInputPath" Options="32768" CheckKernelDebugger="false" CompressionMode="0" VMCodeSectionName=".vmp" VMExecutorCount="1" LicenseDataFileName="" OutputFileName="" WaterMarkName="" RunParameters="">
        <Folders/>
        <Procedures>
            <Procedure MapAddress="VMProtectMarker &quot;xor_transform&quot;" IncludedInCompilation="true" Options="1" CompilationType="0"/>
        </Procedures>
    </Protection>
    <DLLBox/>
    <Script IncludedInCompilation="true"></Script>
</Document>
"@

[System.IO.File]::WriteAllText($projectPath, $projectXml, [System.Text.Encoding]::UTF8)

Write-Host "开始 VMP 保护: $inputPath"
& $vmpCliPath $inputPath $outputPath -pf $projectPath
if ($LASTEXITCODE -ne 0) {
    throw "VMP 保护失败，VMProtect_Con.exe 返回码: $LASTEXITCODE"
}

Write-Host "保护完成: $outputPath"
