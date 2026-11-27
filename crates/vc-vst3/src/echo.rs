//! Held echo, allocated only at plugin initialization. Release invalidates all
//! previous delay contents without scanning or allocating on the audio thread.
pub(crate) struct HoldEcho {
    delay: Vec<[f32; 2]>,
    position: usize,
    valid: usize,
}
impl HoldEcho {
    pub(crate) fn new(rate: u32) -> Self {
        let samples = ((rate as u64 * 180) / 1000).max(1) as usize;
        Self {
            delay: vec![[0.0; 2]; samples],
            position: 0,
            valid: 0,
        }
    }
    pub(crate) fn clear(&mut self) {
        self.position = 0;
        self.valid = 0;
    }
    pub(crate) fn process(&mut self, channels: &mut [&mut [f32]], held: bool) {
        if !held {
            self.clear();
            return;
        }
        let Some((first, rest)) = channels.split_first_mut() else {
            return;
        };
        let mut second = rest.first_mut();
        for (i, left) in first.iter_mut().enumerate() {
            let previous = if self.valid == self.delay.len() {
                self.delay[self.position]
            } else {
                [0.0; 2]
            };
            let dry = *left;
            self.delay[self.position][0] = 0.65 * dry + 0.35 * previous[0];
            *left = 0.78 * dry + 0.22 * previous[0];
            if let Some(right) = second.as_mut() {
                let dry = right[i];
                self.delay[self.position][1] = 0.65 * dry + 0.35 * previous[1];
                right[i] = 0.78 * dry + 0.22 * previous[1];
            } else {
                self.delay[self.position][1] = 0.0;
            }
            self.position = (self.position + 1) % self.delay.len();
            self.valid = (self.valid + 1).min(self.delay.len());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn echo_is_held_only_and_does_not_leak_on_repress() {
        let mut e = HoldEcho::new(1000);
        let mut audio = vec![0.0; 181];
        audio[0] = 1.0;
        e.process(&mut [&mut audio], true);
        assert!((audio[0] - 0.78).abs() < 1e-6);
        assert!(audio[180] > 0.1);
        let mut dry = [0.3, -0.2];
        e.process(&mut [&mut dry], false);
        assert_eq!(dry, [0.3, -0.2]);
        let mut silent = vec![0.0; 500];
        e.process(&mut [&mut silent], true);
        assert!(silent.iter().all(|v| *v == 0.0));
    }
    #[test]
    fn stereo_channels_remain_independent() {
        let mut e = HoldEcho::new(1000);
        let mut l = vec![0.0; 181];
        let mut r = vec![0.0; 181];
        l[0] = 1.0;
        e.process(&mut [&mut l, &mut r], true);
        assert!(l[180] > 0.1);
        assert!(r.iter().all(|v| *v == 0.0));
    }
}
