param([string]$InputWav)
$ErrorActionPreference = 'Stop'
# Run from the installed bundle tools folder, after model conversion.
$bundle = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $bundle 'Contents/x86_64-win'
$exe = Join-Path $bin 'vc-rs.exe'
$metrics = Join-Path $bin 'voicebank_wav_metrics.exe'
$env:VC_RS_TENSORRT_BUILDER_HELPER = Join-Path $bin 'vc-tensorrt-builder.exe'
$env:VC_RS_TENSORRT_CACHE_DIR = Join-Path $bundle 'engine-cache'
$env:PATH = "$bin;$env:PATH"
if (!$InputWav) {
    Add-Type -AssemblyName System.Windows.Forms
    $d = New-Object System.Windows.Forms.OpenFileDialog
    try {
        $d.Filter='WAV|*.wav'; $d.Title='Select one 48 kHz speech WAV (10-20 seconds)'
        if ($d.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return }
        $InputWav=$d.FileName
    } finally { $d.Dispose() }
}
$InputWav=(Resolve-Path -LiteralPath $InputWav).Path
$embedder=Join-Path $bundle 'models/common/contentvec-f.onnx'
$f0=Join-Path $bundle 'models/common/rmvpe_20231006.onnx'
foreach ($file in @($exe,$metrics,$embedder,$f0,$env:VC_RS_TENSORRT_BUILDER_HELPER)) {
    if (!(Test-Path -LiteralPath $file -PathType Leaf)) { throw "Missing local file: $file" }
}
$root=Join-Path $bundle ('measurements/stage03-'+(Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
New-Item -ItemType Directory -Force $root | Out-Null
$raw=& $metrics $InputWav
if ($LASTEXITCODE -ne 0) { throw 'Input WAV metrics failed' }
$raw | Set-Content (Join-Path $root 'input.json') -Encoding UTF8
$info=($raw -join "`n") | ConvertFrom-Json
if ($info.sample_rate -ne 48000 -or $info.nonfinite_samples -ne 0) { throw 'Use a valid 48 kHz WAV, matching Wave Link.' }
Copy-Item -LiteralPath $InputWav (Join-Path $root 'original.wav')
$profiles=@(
    @{slot='toddler';pitch=14;index=0.4;protect=0.25;rms=0.75;pitchAlt=12;formantAlt=2},
    @{slot='girl';pitch=12;index=0.4;protect=0.25;rms=0.75;pitchAlt=10;formantAlt=2},
    @{slot='male';pitch=0;index=0.5;protect=0.33;rms=0.8;pitchAlt=-2;formantAlt=-2},
    @{slot='robot';pitch=0;index=0.7;protect=0.33;rms=1.0;pitchAlt=2;formantAlt=0}
)
$results=@()
$profiles | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'profiles.json') -Encoding UTF8
foreach ($p in $profiles) {
    $slot=$p.slot
    $dir=Join-Path $root $slot
    New-Item -ItemType Directory -Force $dir | Out-Null
    $modelDir=Join-Path $bundle "models/$slot"
    $model=Join-Path $modelDir "$slot.onnx"
    $indexes=@(Get-ChildItem -LiteralPath $modelDir -File -Filter '*.index' -ErrorAction SilentlyContinue)
    if (!(Test-Path -LiteralPath $model) -or $indexes.Count -ne 1) {
        $results += [pscustomobject]@{slot=$slot;variant='all';status='failed';error='Need local ONNX and exactly one INDEX';pitch=$null;index=$null;protect=$null;formant=$null;rms_mix=$null;context_ms=$null;processing_p95_ms=$null;deadline_misses=$null;nonfinite=$null;peak=$null;duration_seconds=$null;wav=''}
        continue
    }
    $index=$indexes[0].FullName
    $variants=@(
        @{name='baseline';pitch=$p.pitch;index=$p.index;protect=$p.protect;formant=0;rms=0.0;context=100},
        @{name='provisional';pitch=$p.pitch;index=$p.index;protect=$p.protect;formant=0;rms=$p.rms;context=100},
        @{name='pitch';pitch=$p.pitchAlt;index=$p.index;protect=$p.protect;formant=0;rms=$p.rms;context=100},
        @{name='index';pitch=$p.pitch;index=[Math]::Max(0,$p.index-0.2);protect=$p.protect;formant=0;rms=$p.rms;context=100},
        @{name='context';pitch=$p.pitch;index=$p.index;protect=$p.protect;formant=0;rms=$p.rms;context=400}
    )
    if ($p.formantAlt -ne 0) { $variants+=@{name='formant';pitch=$p.pitch;index=$p.index;protect=$p.protect;formant=$p.formantAlt;rms=$p.rms;context=100} }
    foreach ($v in $variants) {
        $name=$v.name; $wav=Join-Path $dir "$name.wav";$timing=Join-Path $dir "$name.timing.json";$join=Join-Path $dir "$name.join.csv"
        $row=[ordered]@{slot=$slot;variant=$name;status='failed';error='';pitch=$v.pitch;index=$v.index;protect=$v.protect;formant=$v.formant;rms_mix=$v.rms;context_ms=$v.context;processing_p95_ms=$null;deadline_misses=$null;nonfinite=$null;peak=$null;duration_seconds=$null;wav=$wav}
        try {
            $ci=[System.Globalization.CultureInfo]::InvariantCulture
            $argsList=@('wav','--input',$InputWav,'--output',$wav,'--model',$model,'--embedder',$embedder,'--f0-model',$f0,'--provider','tensorrt','--chunk-ms','50','--crossfade-ms','40','--sola-search-ms','12','--rvc-output-tail-discard-ms','10','--smoother','phase-vocoder',"--pitch-shift=$($v.pitch)","--index-rate=$(([double]$v.index).ToString($ci))",'--index',$index,"--protect=$(([double]$v.protect).ToString($ci))","--formant-shift=$($v.formant)","--rms-mix-rate=$(([double]$v.rms).ToString($ci))",'--extra-convert-ms',"$($v.context)",'--peak-protection','--voice-f0-limits','--performance-report',$timing,'--performance-warmup-chunks','3','--join-report',$join)
            & $exe @argsList 1> (Join-Path $dir "$name.stdout.log") 2> (Join-Path $dir "$name.stderr.log")
            if ($LASTEXITCODE -ne 0) { throw "Conversion exit $LASTEXITCODE" }
            $report=& $metrics $wav
            if ($LASTEXITCODE -ne 0) { throw 'WAV metrics failed' }
            $report | Set-Content (Join-Path $dir "$name.metrics.json") -Encoding UTF8
            $q=($report -join "`n") | ConvertFrom-Json
            $t=Get-Content -LiteralPath $timing -Raw | ConvertFrom-Json
            $row.processing_p95_ms=$t.steady.processing_ms.p95_ms
            $row.deadline_misses=$t.steady.processing_ms.deadline_misses
            $row.nonfinite=$q.nonfinite_samples; $row.peak=$q.peak; $row.duration_seconds=$q.duration_seconds
            if ($q.nonfinite_samples -ne 0 -or [Math]::Abs($q.duration_seconds-$info.duration_seconds) -gt 0.05 -or $q.rms -le 0) { throw 'Invalid, silent or duration-mismatched output' }
            $row.status='measured'
        } catch { $row.error=$_.Exception.Message }
        $results += [pscustomobject]$row
        $results | Export-Csv (Join-Path $root 'summary.csv') -NoTypeInformation -Encoding UTF8
    }
}
$results | Export-Csv (Join-Path $root 'summary.csv') -NoTypeInformation -Encoding UTF8
Write-Host "Results: $root"
if (@($results | Where-Object status -eq 'failed').Count -gt 0) { exit 1 }
