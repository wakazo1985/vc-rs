[CmdletBinding()]
param([Parameter(Mandatory)][string]$Bundle)
$ErrorActionPreference='Stop'
$bundlePath=(Resolve-Path -LiteralPath $Bundle).Path
if (!(Test-Path -LiteralPath $bundlePath -PathType Container)) { throw 'VST3 bundle must be a folder' }
$name=Split-Path -Leaf $bundlePath
if ($name -notlike '*.vst3') { throw 'Bundle name must end in .vst3' }
$module=Join-Path $bundlePath "Contents\x86_64-win\$name"
$bootstrap=Join-Path (Split-Path -Parent $module) 'Microsoft.WindowsAppRuntime.Bootstrap.dll'
function Get-PEMachine([string]$Path) {
    $s=[IO.File]::OpenRead($Path)
    try {
        $b=New-Object IO.BinaryReader($s)
        if ($b.ReadUInt16() -ne 0x5a4d) { throw "Invalid MZ header: $Path" }
        $s.Position=0x3c; $offset=$b.ReadInt32()
        $s.Position=$offset
        if ($b.ReadUInt32() -ne 0x4550) { throw "Invalid PE header: $Path" }
        return $b.ReadUInt16()
    } finally { $s.Dispose() }
}
$files=@()
foreach ($p in @($module,$bootstrap)) {
    if (!(Test-Path -LiteralPath $p -PathType Leaf)) { throw "Missing required file: $p" }
    $machine=Get-PEMachine $p
    if ($machine -ne 0x8664) { throw "Not Windows x64: $p" }
    $files += [ordered]@{name=(Split-Path -Leaf $p);machine=('0x{0:X4}' -f $machine);sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash}
}
$probe=Join-Path $bundlePath ('write-test-'+[Guid]::NewGuid().ToString('N')+'.tmp')
$writeOk=$false; $writeError=$null
try {
    $stream=[IO.File]::Open($probe,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
    try { $stream.WriteByte(1); $stream.Flush() } finally { $stream.Dispose() }
    Remove-Item -LiteralPath $probe -ErrorAction Stop
    $writeOk=$true
} catch { $writeError=$_.Exception.Message }
$runtimes=@()
try { $runtimes=@(Get-AppxPackage '*WindowsAppRuntime*' -ErrorAction Stop | Select-Object Name,Version,Architecture) } catch {}
$wave=@(Get-Process -ErrorAction SilentlyContinue | Where-Object {$_.ProcessName -match '^WaveLink|^Wave Link'} | Select-Object ProcessName,Id)
$result=[ordered]@{bundle=$bundlePath;files=$files;bundle_writable=$writeOk;write_error=$writeError;windows_app_runtime_packages=$runtimes;wave_link_processes=$wave;scope='Structural and current-user write checks only, not a Wave Link audio test.'}
$result | ConvertTo-Json -Depth 6
if (!$writeOk) { throw 'VST3 folder is not writable by the current user; settings/logs cannot be saved. No AppData fallback.' }
