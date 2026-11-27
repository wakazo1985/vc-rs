$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
$exe = Join-Path $PSScriptRoot 'vc-rs.exe'
$metrics = Join-Path $PSScriptRoot 'voicebank_wav_metrics.exe'
function Select-File([string]$title,[string]$filter) {
    $d = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $d.Title=$title; $d.Filter=$filter; $d.CheckFileExists=$true
        if ($d.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { throw "Cancelled: $title" }
        return $d.FileName
    } finally { $d.Dispose() }
}
function Run-Native([string]$binary,[string[]]$arguments,[string]$stdout,[string]$stderr) {
    $quoted = @($arguments | ForEach-Object { '"' + $_ + '"' })
    $p = Start-Process -FilePath $binary -ArgumentList $quoted -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    return $p.ExitCode
}
if (!(Test-Path -LiteralPath $exe) -or !(Test-Path -LiteralPath $metrics)) { throw 'Benchmark binaries missing' }
$input = Select-File 'Select the same speech WAV for all voices (10-30 seconds recommended)' 'WAV|*.wav'
$embedder = Select-File 'Select local ContentVec ONNX (common embedder)' 'ONNX|*.onnx'
$f0 = Select-File 'Select local RMVPE ONNX (common F0 model)' 'ONNX|*.onnx'
$provider = 'windowsml-directml'
$chunk = 500
$root = Join-Path $PSScriptRoot ('measurements\run-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Path $root -Force | Out-Null
$results = @()
$wav = Join-Path $root 'original.wav'
Copy-Item -LiteralPath $input -Destination $wav
if ((Run-Native $metrics @($wav) (Join-Path $root 'original.metrics.json') (Join-Path $root 'original.metrics.log')) -ne 0) { throw 'Input WAV analysis failed' }
$manifest = [ordered]@{
    started_utc=[DateTime]::UtcNow.ToString('o')
    provider_requested=$provider
    chunk_ms=$chunk
    input_sha256=(Get-FileHash -LiteralPath $input -Algorithm SHA256).Hash
    embedder_sha256=(Get-FileHash -LiteralPath $embedder -Algorithm SHA256).Hash
    f0_sha256=(Get-FileHash -LiteralPath $f0 -Algorithm SHA256).Hash
    gpu=@(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name,DriverVersion)
    cpu=@(Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
    scope='Offline per-model processing. No claim of Wave Link realtime behavior, GPU placement, VRAM or perceptual quality.'
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $root 'environment.json') -Encoding UTF8
foreach ($slot in @('toddler','girl','male','robot')) {
    Write-Host "Testing $slot"
    try {
        $model = Select-File "Select $slot ONNX" 'ONNX|*.onnx'
        $hash = (Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash
        $dir = Join-Path $root $slot
        New-Item -ItemType Directory -Path $dir | Out-Null
        for ($run=1; $run -le 3; $run++) {
            $out=Join-Path $dir "run-$run.wav"
            $timing=Join-Path $dir "run-$run.timing.json"
            $join=Join-Path $dir "run-$run.join.csv"
            $argsList=@('wav','--input',$wav,'--output',$out,'--model',$model,'--embedder',$embedder,'--f0-model',$f0,'--provider',$provider,'--chunk-ms',"$chunk",'--smoother','phase-vocoder','--extra-convert-ms','100','--rms-mix-rate','0','--performance-report',$timing,'--performance-warmup-chunks','3','--join-report',$join)
            $exitCode=Run-Native $exe $argsList (Join-Path $dir "run-$run.stdout.log") (Join-Path $dir "run-$run.stderr.log")
            if ($exitCode -ne 0) { throw "Conversion failed on run $run (exit $exitCode)" }
            $metricPath=Join-Path $dir "run-$run.metrics.json"
            if ((Run-Native $metrics @($out) $metricPath (Join-Path $dir "run-$run.metrics.log")) -ne 0) { throw "WAV analysis failed on run $run" }
            $stats=Get-Content -LiteralPath $metricPath -Raw | ConvertFrom-Json
            $t=Get-Content -LiteralPath $timing -Raw | ConvertFrom-Json
            $p=$t.steady.processing_ms
            $results += [pscustomobject]@{slot=$slot;run=$run;status='measured';model_sha256=$hash;processing_p95_ms=$p.p95_ms;processing_max_ms=$p.max_ms;deadline_misses=$p.deadline_misses;steady_chunks=$p.count;rms=$stats.rms;peak=$stats.peak;nonfinite=$stats.nonfinite_samples;near_full_scale_fraction=$stats.near_full_scale_fraction;max_sample_step=$stats.max_adjacent_sample_step;duration_seconds=$stats.duration_seconds;error=''}
        }
    } catch {
        $results += [pscustomobject]@{slot=$slot;run=0;status='failed';model_sha256='';processing_p95_ms=$null;processing_max_ms=$null;deadline_misses=$null;steady_chunks=$null;rms=$null;peak=$null;nonfinite=$null;near_full_scale_fraction=$null;max_sample_step=$null;duration_seconds=$null;error=$_.Exception.Message}
        Write-Warning "$slot : $($_.Exception.Message)"
    }
    $results | Export-Csv -LiteralPath (Join-Path $root 'summary.csv') -NoTypeInformation -Encoding UTF8
}
Write-Host "Results: $root"
if (@($results | Where-Object status -eq 'failed').Count -gt 0) { exit 1 }
