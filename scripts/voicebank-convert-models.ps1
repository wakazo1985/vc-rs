$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.IO.Compression.FileSystem
$exe = Join-Path $PSScriptRoot 'voicebank_convert.exe'
$root = Join-Path $env:LOCALAPPDATA 'RVCVoiceBank'
$models = Join-Path $root 'models'
New-Item -ItemType Directory -Force -Path $models | Out-Null
$log = Join-Path $root ('conversion-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$results = @()
function Log([string]$message) {
    $line = ('{0} {1}' -f (Get-Date -Format 'HH:mm:ss'), $message)
    Add-Content -LiteralPath $log -Value $line -Encoding UTF8
    Write-Host $line
}
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw 'voicebank_convert.exe is missing.' }
$slots = @(
    @{ Key='toddler'; Name='幼児 / Maxwell Young' },
    @{ Key='girl'; Name='少女 / あみたろ・るんるん' },
    @{ Key='male'; Name='低音男性 / D.pth' },
    @{ Key='robot'; Name='ロボット / Speak & Spell' }
)
foreach ($slot in $slots) {
    $work = $null
    $output = Join-Path $models ($slot.Key + '.onnx')
    try {
        Log ("SELECT: " + $slot.Name)
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        try {
            $dialog.Title = $slot.Name + ' : select PTH or ZIP'
            $dialog.Filter = 'RVC model (*.pth;*.zip)|*.pth;*.zip'
            $dialog.CheckFileExists = $true
            if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
                throw 'Selection cancelled'
            }
            $inputPath = $dialog.FileName
        } finally { $dialog.Dispose() }
        if (Test-Path -LiteralPath $output) { throw "Output exists; move it before reconverting: $output" }
        if ([IO.Path]::GetExtension($inputPath) -ieq '.zip') {
            $work = Join-Path $env:TEMP ('voicebank-' + [guid]::NewGuid().ToString('N'))
            New-Item -ItemType Directory -Path $work | Out-Null
            $zip = [IO.Compression.ZipFile]::OpenRead($inputPath)
            try {
                $entries = @($zip.Entries | Where-Object { $_.Name -and [IO.Path]::GetExtension($_.Name) -ieq '.pth' -and $_.FullName -notmatch '(^|/)__MACOSX/' })
                if ($entries.Count -eq 0) { throw 'No .pth found in ZIP' }
                $n = 0
                if ($entries.Count -gt 1) {
                    for ($i=0; $i -lt $entries.Count; $i++) { Write-Host ("[{0}] {1} ({2} bytes)" -f ($i+1), $entries[$i].FullName, $entries[$i].Length) }
                    $choice = Read-Host 'Select model number'
                    $parsed = 0
                    if (-not [int]::TryParse($choice,[ref]$parsed) -or $parsed -lt 1 -or $parsed -gt $entries.Count) { throw 'Invalid selection' }
                    $n = $parsed - 1
                }
                $entry = $entries[$n]
                Log ("ZIP entry: " + $entry.FullName)
                $inputPath = Join-Path $work 'model.pth'
                [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$inputPath,$false)
            } finally { $zip.Dispose() }
        }
        Log ("Converting " + $slot.Key)
        # Capture stdout and stderr separately without treating native stderr as a PowerShell error.
        $stdOut = Join-Path $root ($slot.Key + '.stdout.log')
        $stdErr = Join-Path $root ($slot.Key + '.stderr.log')
        $process = Start-Process -FilePath $exe -ArgumentList @(('"' + $inputPath + '"'),('"' + $output + '"')) -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdOut -RedirectStandardError $stdErr
        foreach ($file in @($stdOut,$stdErr)) {
            Get-Content -LiteralPath $file | ForEach-Object { Log $_ }
        }
        if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $output -PathType Leaf)) { throw "Converter failed: exit $($process.ExitCode)" }
        $sha = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
        Log ("SUCCESS: " + $slot.Key + " SHA256=" + $sha)
        $results += [pscustomobject]@{ slot=$slot.Key; status='success'; path=$output; sha256=$sha }
    } catch {
        Log ("FAILED: " + $slot.Key + " : " + $_.Exception.Message)
        $results += [pscustomobject]@{ slot=$slot.Key; status='failed'; error=$_.Exception.Message }
    } finally {
        if ($work -and (Test-Path -LiteralPath $work)) { Remove-Item -LiteralPath $work -Recurse -Force }
    }
}
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $root 'conversion-results.json') -Encoding UTF8
$results | Format-Table slot,status,path -AutoSize
Log ("Results: " + (Join-Path $root 'conversion-results.json'))
Log ("Log: " + $log)
if (@($results | Where-Object status -ne 'success').Count -gt 0) { exit 1 }
exit 0
