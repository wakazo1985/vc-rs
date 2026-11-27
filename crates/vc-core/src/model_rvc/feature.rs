use anyhow::{bail, Context, Result};

#[derive(Default)]
pub(super) struct FeatureTensor {
    pub(super) data: Vec<f32>,
    pub(super) shape: Vec<i64>,
}

/// RVC's unvoiced protection: preserve original ContentVec features when
/// F0 is absent; voiced frames retain the full retrieval blend. Both tensors
/// must already use the identical 10 ms feature grid.
pub(super) fn protect_unvoiced(mixed: &mut [f32], original: &[f32], pitchf: &[f32], channels: usize, protect: f32) -> Result<()> {
    anyhow::ensure!(channels > 0 && mixed.len() == original.len() && mixed.len() == pitchf.len().saturating_mul(channels), "Protect frame alignment mismatch");
    anyhow::ensure!(protect.is_finite() && (0.0..=0.5).contains(&protect), "Invalid Protect");
    if protect >= 0.5 { return Ok(()); }
    for ((target, source), f0) in mixed.chunks_exact_mut(channels).zip(original.chunks_exact(channels)).zip(pitchf) {
        if *f0 < 1.0 {
            for (x, y) in target.iter_mut().zip(source) { *x = protect * *x + (1.0-protect) * *y; }
        }
    }
    Ok(())
}

impl FeatureTensor {
    pub(super) fn repeat_frames(&mut self, factor: usize) -> Result<()> {
        if factor <= 1 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        // Repeat each frame `factor` times in place to reuse the buffer
        // allocated by `extract` instead of building a fresh Vec every chunk.
        // Walk frames back-to-front: frame `f`'s destination blocks start at
        // `f * factor * channels >= f * channels`, so a backward pass never
        // overwrites a source frame that has not been copied yet.
        let old_len = self.data.len();
        self.data.resize(old_len * factor, 0.0);
        for frame in (0..frames).rev() {
            let src = frame * channels;
            for repeat in (0..factor).rev() {
                let dst = (frame * factor + repeat) * channels;
                self.data.copy_within(src..src + channels, dst);
            }
        }
        self.shape[1] = (frames * factor) as i64;
        Ok(())
    }

    pub(super) fn trim_front_frames(&mut self, frames_to_drop: usize) -> Result<()> {
        if frames_to_drop == 0 {
            return Ok(());
        }
        if self.shape.len() != 3 {
            bail!("feature tensor must be rank-3 [1, frames, channels]");
        }
        let batch = usize::try_from(self.shape[0]).context("invalid feature batch")?;
        let frames = usize::try_from(self.shape[1]).context("invalid feature frames")?;
        let channels = usize::try_from(self.shape[2]).context("invalid feature channels")?;
        if batch != 1 {
            bail!("feature batch must be 1, got {batch}");
        }
        if frames_to_drop >= frames {
            return Ok(());
        }
        let sample_offset = frames_to_drop * channels;
        self.data.drain(..sample_offset);
        self.shape[1] = (frames - frames_to_drop) as i64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protect_preserves_voiced_and_restores_unvoiced_on_aligned_grid() {
        let original = [1.0,2.0,3.0,4.0];
        let mut mixed = [5.0,6.0,7.0,8.0];
        protect_unvoiced(&mut mixed,&original,&[220.0,0.0],2,0.25).unwrap();
        assert_eq!(mixed,[5.0,6.0,4.0,5.0]);
        let mut disabled = [5.0,6.0,7.0,8.0];
        protect_unvoiced(&mut disabled,&original,&[220.0,0.0],2,0.5).unwrap();
        assert_eq!(disabled,[5.0,6.0,7.0,8.0]);
        assert!(protect_unvoiced(&mut disabled,&original,&[0.0],2,0.25).is_err());
    }

    #[test]
    fn repeat_frames_duplicates_each_frame_in_order() {
        // 3 frames x 2 channels.
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape: vec![1, 3, 2],
        };
        tensor.repeat_frames(2).unwrap();
        assert_eq!(
            tensor.data,
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(tensor.shape, vec![1, 6, 2]);
    }

    #[test]
    fn repeat_frames_factor_three() {
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0],
            shape: vec![1, 1, 2],
        };
        tensor.repeat_frames(3).unwrap();
        assert_eq!(tensor.data, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
        assert_eq!(tensor.shape, vec![1, 3, 2]);
    }

    #[test]
    fn repeat_frames_factor_one_is_noop() {
        let mut tensor = FeatureTensor {
            data: vec![1.0, 2.0, 3.0, 4.0],
            shape: vec![1, 2, 2],
        };
        tensor.repeat_frames(1).unwrap();
        assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.shape, vec![1, 2, 2]);
    }
}
