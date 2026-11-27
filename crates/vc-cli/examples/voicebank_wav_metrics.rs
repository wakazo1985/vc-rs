//! Offline WAV metrics. These numbers do not measure perceived naturalness.
use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader};
use serde_json::{json, Value};
use std::{
    io::{Read, Seek},
    path::Path,
};

fn measure<R: Read + Seek>(mut reader: WavReader<R>) -> Result<Value> {
    let spec = reader.spec();
    let channels = usize::from(spec.channels);
    anyhow::ensure!(channels > 0 && spec.sample_rate > 0, "Invalid WAV format");
    let mut previous = vec![None::<f64>; channels];
    let mut count = 0u64;
    let mut finite = 0u64;
    let mut zero = 0u64;
    let mut clipped = 0u64;
    let mut peak = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum = 0.0f64;
    let mut max_step = 0.0f64;
    let mut add = |value: f64| {
        let ch = count as usize % channels;
        count += 1;
        if !value.is_finite() {
            previous[ch] = None;
            return;
        }
        finite += 1;
        let magnitude = value.abs();
        peak = peak.max(magnitude);
        sum_sq += value * value;
        sum += value;
        if magnitude <= 1e-6 {
            zero += 1;
        }
        if magnitude >= 0.999 {
            clipped += 1;
        }
        if let Some(before) = previous[ch] {
            max_step = max_step.max((value - before).abs());
        }
        previous[ch] = Some(value);
    };
    match spec.sample_format {
        SampleFormat::Float => {
            for value in reader.samples::<f32>() {
                add(f64::from(value?));
            }
        }
        SampleFormat::Int => {
            anyhow::ensure!(
                (1..=32).contains(&spec.bits_per_sample),
                "Invalid PCM bit depth"
            );
            let scale = (1u64 << (spec.bits_per_sample - 1)) as f64;
            for value in reader.samples::<i32>() {
                add(f64::from(value?) / scale);
            }
        }
    }
    anyhow::ensure!(count > 0, "Empty WAV");
    let denominator = finite.max(1) as f64;
    Ok(json!({
        "sample_rate":spec.sample_rate,"channels":channels,
        "duration_seconds":count as f64 / channels as f64 / spec.sample_rate as f64,
        "samples":count,"nonfinite_samples":count-finite,
        "rms":(sum_sq/denominator).sqrt(),"peak":peak,"dc_mean":sum/denominator,
        "near_zero_fraction":zero as f64/denominator,
        "near_full_scale_fraction":clipped as f64/denominator,
        "max_adjacent_sample_step":max_step,
        "scope":"Signal-level checks only; does not establish voice naturalness, pitch accuracy, audible clipping or distortion."
    }))
}
fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("usage: voicebank_wav_metrics file.wav")?;
    let reader = WavReader::open(Path::new(&path))?;
    println!("{}", serde_json::to_string_pretty(&measure(reader)?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_stereo_signal_does_not_compare_between_channels() {
        let mut data = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48000,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        {
            let mut w = hound::WavWriter::new(&mut data, spec).unwrap();
            for _ in 0..32 {
                w.write_sample(16384i16).unwrap();
                w.write_sample(-16384i16).unwrap();
            }
            w.finalize().unwrap();
        }
        data.set_position(0);
        let report = measure(WavReader::new(data).unwrap()).unwrap();
        assert_eq!(report["rms"], 0.5);
        assert_eq!(report["peak"], 0.5);
        assert_eq!(report["max_adjacent_sample_step"], 0.0);
        assert_eq!(report["nonfinite_samples"], 0);
    }
}
