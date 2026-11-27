//! Spectral-envelope formant adjustment on a complete RVC output candidate.
//! Cepstral smoothing separates the spectral envelope from harmonic detail;
//! warp only the envelope, retain complex spectral phase and use normalized
//! overlap-add. No resampling or changes to TensorRT input/output dimensions.
use anyhow::{ensure, Result};
use realfft::{num_complex::Complex32, ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub(super) struct Formant {
    n: usize,
    rate: u32,
    semitones: f32,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    bins: Vec<Complex32>,
    env: Vec<Complex32>,
    cep: Vec<f32>,
    scratch: Vec<Complex32>,
    window: Vec<f32>,
    sum: Vec<f32>,
    norm: Vec<f32>,
}
impl Formant {
    pub(super) fn new(rate: u32, semitones: f32) -> Result<Self> {
        ensure!(
            rate >= 16000 && semitones.is_finite() && (-12.0..=12.0).contains(&semitones),
            "invalid formant configuration"
        );
        let n = ((rate as usize * 32 / 1000).next_power_of_two()).max(512);
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(n);
        let inverse = planner.plan_fft_inverse(n);
        let scratch =
            vec![Complex32::default(); forward.get_scratch_len().max(inverse.get_scratch_len())];
        Ok(Self {
            n,
            rate,
            semitones,
            time: forward.make_input_vec(),
            bins: forward.make_output_vec(),
            env: forward.make_output_vec(),
            cep: inverse.make_output_vec(),
            scratch,
            window: (0..n)
                .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / n as f32).sin())
                .collect(),
            forward,
            inverse,
            sum: Vec::new(),
            norm: Vec::new(),
        })
    }
    pub(super) fn process(&mut self, audio: &mut [f32]) -> Result<()> {
        if self.semitones == 0.0 || audio.is_empty() {
            return Ok(());
        }
        ensure!(audio.iter().all(|x| x.is_finite()), "invalid formant input");
        let n = self.n;
        let hop = n / 4;
        self.sum.resize(audio.len(), 0.0);
        self.sum.fill(0.0);
        self.norm.resize(audio.len(), 0.0);
        self.norm.fill(0.0);
        let factor = 2.0_f32.powf(self.semitones / 12.0);
        // Smooth at ~1.5 ms quefrency; suppress most excitation-period detail.
        let cutoff = self.rate as f32 * 0.0015;
        let mut start = -(n as isize / 2);
        while start < audio.len() as isize {
            for i in 0..n {
                let at = start + i as isize;
                self.time[i] = if at >= 0 && (at as usize) < audio.len() {
                    audio[at as usize] * self.window[i]
                } else {
                    0.0
                };
            }
            self.forward
                .process_with_scratch(&mut self.time, &mut self.bins, &mut self.scratch)?;
            let peak = self.bins.iter().map(|z| z.norm()).fold(0.0_f32, f32::max);
            if peak > 1e-8 {
                let floor = (peak * 1e-5).max(1e-12);
                for (e, b) in self.env.iter_mut().zip(&self.bins) {
                    *e = Complex32::new(b.norm().max(floor).ln(), 0.0);
                }
                self.inverse.process_with_scratch(
                    &mut self.env,
                    &mut self.cep,
                    &mut self.scratch,
                )?;
                for (i, v) in self.cep.iter_mut().enumerate() {
                    let q = i.min(n - i) as f32 / cutoff;
                    *v *= (-0.5 * q * q).exp() / n as f32;
                }
                self.forward.process_with_scratch(
                    &mut self.cep,
                    &mut self.env,
                    &mut self.scratch,
                )?;
                for (k, b) in self.bins.iter_mut().enumerate() {
                    let at = (k as f32 / factor).min((self.env.len() - 1) as f32);
                    let lo = at.floor() as usize;
                    let hi = (lo + 1).min(self.env.len() - 1);
                    let target =
                        self.env[lo].re + (self.env[hi].re - self.env[lo].re) * (at - lo as f32);
                    // Bound envelope gains to avoid runaway peaks on weak harmonics.
                    let gain = (target - self.env[k].re).clamp(-2.0, 2.0).exp();
                    *b *= gain;
                }
            }
            self.inverse
                .process_with_scratch(&mut self.bins, &mut self.time, &mut self.scratch)?;
            for i in 0..n {
                let at = start + i as isize;
                if at >= 0 && (at as usize) < audio.len() {
                    let at = at as usize;
                    let w = self.window[i];
                    self.sum[at] += self.time[i] * w / n as f32;
                    self.norm[at] += w * w;
                }
            }
            start += hop as isize;
        }
        for (i, v) in audio.iter_mut().enumerate() {
            *v = self.sum[i] / self.norm[i].max(1e-8);
        }
        ensure!(
            audio.iter().all(|x| x.is_finite()),
            "invalid formant output"
        );
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn formant_zero_is_exact_bypass_and_silence_remains_zero() {
        let mut samples = vec![0.2, -0.3, 0.5, 0.0];
        let original = samples.clone();
        Formant::new(48000, 0.0)
            .unwrap()
            .process(&mut samples)
            .unwrap();
        assert_eq!(samples, original);
        let mut silence = vec![0.0; 2000];
        Formant::new(40000, 4.0)
            .unwrap()
            .process(&mut silence)
            .unwrap();
        assert!(silence.iter().all(|x| *x == 0.0));
    }
    fn energy(x: &[f32], hz: f32, rate: u32) -> f64 {
        let (a, b) = x.iter().enumerate().fold((0.0, 0.0), |(a, b), (i, v)| {
            let p = std::f64::consts::TAU * hz as f64 * i as f64 / rate as f64;
            (a + *v as f64 * p.cos(), b + *v as f64 * p.sin())
        });
        a * a + b * b
    }
    #[test]
    fn formant_moves_envelope_without_resampling_harmonic_grid() {
        for rate in [40000u32, 48000] {
            let n = rate as usize / 5;
            let input: Vec<f32> = (0..n)
                .map(|i| {
                    (1..=40)
                        .map(|h| {
                            let hz = h as f32 * 100.0;
                            let amp = 0.05 + (-0.5 * ((hz - 1000.0) / 250.0).powi(2)).exp();
                            amp * 0.02 * (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin()
                        })
                        .sum()
                })
                .collect();
            let ratio = |x: &[f32]| {
                let interior = &x[rate as usize / 20..rate as usize * 3 / 20];
                energy(interior, 1500.0, rate) / energy(interior, 1000.0, rate).max(1e-12)
            };
            let before = ratio(&input);
            let mut shifted = input.clone();
            Formant::new(rate, 7.0)
                .unwrap()
                .process(&mut shifted)
                .unwrap();
            assert_eq!(shifted.len(), input.len());
            assert!(shifted.iter().all(|x| x.is_finite()));
            assert!(
                ratio(&shifted) > before * 1.5,
                "rate={rate} before={before} after={}",
                ratio(&shifted)
            );
            let middle = &shifted[rate as usize / 20..rate as usize * 3 / 20];
            assert!(energy(middle, 1500.0, rate) > energy(middle, 1550.0, rate) * 10.0);
            let mut down = input.clone();
            Formant::new(rate, -7.0)
                .unwrap()
                .process(&mut down)
                .unwrap();
            let mid = &down[rate as usize / 20..rate as usize * 3 / 20];
            let orig = &input[rate as usize / 20..rate as usize * 3 / 20];
            assert!(
                energy(mid, 700.0, rate) / energy(mid, 1000.0, rate)
                    > energy(orig, 700.0, rate) / energy(orig, 1000.0, rate)
            );
        }
    }
}
