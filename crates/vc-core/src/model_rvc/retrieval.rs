//! Read the documented 64-bit little-endian FAISS FlatL2 / IVFFlat L2 layout.
//! Format reference: facebookresearch/faiss v1.8.0 faiss/impl/index_write.cpp,
//! io_macros.h (MIT). Search and blending implemented independently.
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
const MAX_BYTES: u64 = 512 * 1024 * 1024;
const MAX_LISTS: usize = 65536;
const K: usize = 8;
struct Reader<R> {
    input: R,
    left: u64,
}
impl<R: Read> Reader<R> {
    fn take(&mut self, out: &mut [u8]) -> Result<()> {
        ensure!(out.len() as u64 <= self.left, "truncated INDEX");
        self.input.read_exact(out)?;
        self.left -= out.len() as u64;
        Ok(())
    }
    fn tag(&mut self) -> Result<[u8; 4]> {
        let mut b = [0; 4];
        self.take(&mut b)?;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        let mut b = [0; 1];
        self.take(&mut b)?;
        Ok(b[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.tag()?))
    }
    fn count(&mut self) -> Result<usize> {
        let mut b = [0; 8];
        self.take(&mut b)?;
        Ok(usize::try_from(u64::from_le_bytes(b))?)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        let mut buf = [0; 8192];
        let mut left = n;
        while left > 0 {
            let amount = left.min(buf.len());
            self.take(&mut buf[..amount])?;
            left -= amount;
        }
        Ok(())
    }
    fn floats(&mut self, n: usize) -> Result<Vec<f32>> {
        let bytes = n.checked_mul(4).context("INDEX float count overflow")?;
        ensure!(bytes as u64 <= self.left, "truncated INDEX vector");
        let mut values = Vec::new();
        values.try_reserve_exact(n)?;
        let mut remaining = n;
        let mut buf = [0; 16384];
        while remaining > 0 {
            let count = remaining.min(buf.len() / 4);
            self.take(&mut buf[..count * 4])?;
            for b in buf[..count * 4].as_chunks::<4>().0 {
                let x = f32::from_le_bytes(*b);
                ensure!(x.is_finite(), "INDEX contains nonfinite value");
                values.push(x);
            }
            remaining -= count;
        }
        Ok(values)
    }
    fn header(&mut self) -> Result<(usize, usize)> {
        let d = self.u32()? as usize;
        let rows = self.count()?;
        ensure!((1..=4096).contains(&d), "unsupported INDEX dimension");
        self.skip(16)?; // historical dummy fields
        ensure!(self.u8()? == 1, "INDEX is not trained");
        ensure!(self.u32()? == 1, "only L2 INDEX metric supported");
        Ok((d, rows))
    }
    fn flat(&mut self) -> Result<(usize, usize, Vec<f32>)> {
        ensure!(self.tag()? == *b"IxF2", "expected FlatL2 quantizer");
        let (d, rows) = self.header()?;
        let n = self.count()?;
        ensure!(
            Some(n) == d.checked_mul(rows),
            "invalid FlatL2 vector count"
        );
        Ok((d, rows, self.floats(n)?))
    }
}
pub(super) struct RetrievalIndex {
    dim: usize,
    centroids: Vec<f32>,
    lists: Vec<Vec<f32>>,
    probe: usize,
}
impl RetrievalIndex {
    pub(super) fn dimension(&self) -> usize {
        self.dim
    }
    pub(super) fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open INDEX {}", path.display()))?;
        let len = file.metadata()?.len();
        ensure!(len <= MAX_BYTES, "INDEX exceeds 512 MiB limit");
        Self::read(BufReader::new(file), len)
    }
    fn read<R: Read>(source: R, len: u64) -> Result<Self> {
        let mut r = Reader {
            input: source,
            left: len,
        };
        let tag = r.tag()?;
        let (d, total) = r.header()?;
        ensure!(total > 0, "INDEX has no added feature vectors");
        ensure!(
            total
                .checked_mul(d)
                .and_then(|n| n.checked_mul(4))
                .is_some_and(|n| n as u64 <= r.left),
            "invalid INDEX row count"
        );
        if tag == *b"IxF2" {
            let n = r.count()?;
            ensure!(
                Some(n) == d.checked_mul(total),
                "invalid FlatL2 vector count"
            );
            let data = r.floats(n)?;
            ensure!(r.left == 0, "unexpected trailing INDEX data");
            return Ok(Self {
                dim: d,
                centroids: Vec::new(),
                lists: vec![data],
                probe: 1,
            });
        }
        ensure!(tag == *b"IwFl", "unsupported INDEX type: {:?}", tag);
        let lists = r.count()?;
        let probe = r.count()?;
        ensure!(
            (1..=MAX_LISTS).contains(&lists) && probe > 0 && probe <= lists,
            "invalid IVF list/probe count"
        );
        let (qd, qn, centroids) = r.flat()?;
        ensure!(qd == d && qn == lists, "invalid IVF quantizer dimensions");
        let direct_map = r.u8()?;
        let map_size = r.count()?;
        r.skip(map_size.checked_mul(8).context("direct map overflow")?)?;
        match direct_map {
            0 | 1 => {}
            2 => {
                let n = r.count()?;
                r.skip(n.checked_mul(16).context("hash map overflow")?)?;
            }
            _ => bail!("unsupported direct map"),
        }
        ensure!(r.tag()? == *b"ilar", "only array inverted lists supported");
        ensure!(
            r.count()? == lists && r.count()? == d * 4,
            "invalid inverted list dimensions"
        );
        let mode = r.tag()?;
        let n = r.count()?;
        let mut sizes = vec![0usize; lists];
        if mode == *b"full" {
            ensure!(n == lists, "invalid full list size table");
            for size in &mut sizes {
                *size = r.count()?;
            }
        } else if mode == *b"sprs" {
            ensure!(
                n % 2 == 0 && n <= lists * 2,
                "invalid sparse list size table"
            );
            for _ in 0..n / 2 {
                let i = r.count()?;
                let size = r.count()?;
                ensure!(
                    i < lists && sizes[i] == 0 && size > 0,
                    "invalid sparse list entry"
                );
                sizes[i] = size;
            }
        } else {
            bail!("unsupported inverted list size table");
        }
        let count = sizes
            .iter()
            .try_fold(0usize, |a, &b| a.checked_add(b))
            .context("INDEX list count overflow")?;
        ensure!(count == total, "INDEX total differs from list sizes");
        let mut data = Vec::with_capacity(lists);
        for count in sizes {
            data.push(r.floats(count.checked_mul(d).context("INDEX list size overflow")?)?);
            // IDs are not needed: the result is read directly from the matching
            // stored vector, not by assuming IDs are contiguous or ordered.
            r.skip(count.checked_mul(8).context("INDEX ids overflow")?)?;
        }
        ensure!(r.left == 0, "unexpected trailing INDEX data");
        // Bound worst-case live search; normal RVC IVF files use nprobe=1.
        Ok(Self {
            dim: d,
            centroids,
            lists: data,
            probe: probe.min(8),
        })
    }
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let z = x - y;
                z * z
            })
            .sum()
    }
    pub(super) fn blend(&self, features: &mut [f32], rate: f32) -> Result<()> {
        ensure!(
            features.len().is_multiple_of(self.dim),
            "INDEX features not frame aligned"
        );
        if rate == 0.0 {
            return Ok(());
        }
        let mut ranked = Vec::<(f32, usize)>::with_capacity(self.lists.len());
        let mut nearest = Vec::<(f32, usize, usize)>::with_capacity(K + 1);
        let mut mix = vec![0.0f32; self.dim];
        for frame in features.chunks_exact_mut(self.dim) {
            if frame.iter().any(|x| !x.is_finite()) {
                continue;
            }
            ranked.clear();
            nearest.clear();
            if self.centroids.is_empty() {
                ranked.push((0.0, 0));
            } else {
                ranked.extend(
                    self.centroids
                        .chunks_exact(self.dim)
                        .enumerate()
                        .map(|(i, c)| (Self::distance(frame, c), i)),
                );
                let probe = self.probe.min(ranked.len());
                if probe < ranked.len() {
                    ranked.select_nth_unstable_by(probe, |a, b| a.0.total_cmp(&b.0));
                    ranked.truncate(probe);
                }
            }
            for &(_, list) in &ranked {
                for (row, v) in self.lists[list].chunks_exact(self.dim).enumerate() {
                    let distance = Self::distance(frame, v);
                    if !distance.is_finite() {
                        continue;
                    }
                    if nearest.len() == K && distance >= nearest[K - 1].0 {
                        continue;
                    }
                    let pos = nearest.partition_point(|x| x.0 <= distance);
                    nearest.insert(pos, (distance, list, row));
                    if nearest.len() > K {
                        nearest.pop();
                    }
                }
            }
            if nearest.is_empty() {
                continue;
            }
            mix.fill(0.0);
            let base = nearest[0].0.max(1e-12);
            let mut sum = 0.0;
            for &(distance, list, row) in &nearest {
                let ratio = base / distance.max(1e-12);
                let weight = ratio * ratio;
                let vector = &self.lists[list][row * self.dim..(row + 1) * self.dim];
                for (m, &x) in mix.iter_mut().zip(vector) {
                    *m += weight * x;
                }
                sum += weight;
            }
            if sum > 0.0 {
                for (x, &m) in frame.iter_mut().zip(&mix) {
                    *x = (1.0 - rate) * *x + rate * (m / sum);
                }
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn u64s(b: &mut Vec<u8>, v: u64) {
        b.extend(v.to_le_bytes());
    }
    fn header(b: &mut Vec<u8>, tag: &[u8; 4], d: u32, n: u64) {
        b.extend(tag);
        b.extend(d.to_le_bytes());
        u64s(b, n);
        u64s(b, 1 << 20);
        u64s(b, 1 << 20);
        b.push(1);
        b.extend(1u32.to_le_bytes());
    }
    fn floats(b: &mut Vec<u8>, v: &[f32]) {
        for x in v {
            b.extend(x.to_le_bytes());
        }
    }
    fn flat(b: &mut Vec<u8>, v: &[f32]) {
        header(b, b"IxF2", 2, (v.len() / 2) as u64);
        u64s(b, v.len() as u64);
        floats(b, v);
    }
    fn ivf() -> Vec<u8> {
        let mut b = Vec::new();
        header(&mut b, b"IwFl", 2, 2);
        u64s(&mut b, 2);
        u64s(&mut b, 1);
        flat(&mut b, &[0.0, 0.0, 10.0, 10.0]);
        b.push(0);
        u64s(&mut b, 0);
        b.extend(b"ilar");
        u64s(&mut b, 2);
        u64s(&mut b, 8);
        b.extend(b"full");
        u64s(&mut b, 2);
        u64s(&mut b, 1);
        u64s(&mut b, 1);
        floats(&mut b, &[1.0, 1.0]);
        u64s(&mut b, 123);
        floats(&mut b, &[11.0, 11.0]);
        u64s(&mut b, 456);
        b
    }
    #[test]
    fn flat_blends_and_zero_rate_bypasses() {
        let mut b = Vec::new();
        flat(&mut b, &[2.0, 4.0]);
        let index = RetrievalIndex::read(&b[..], b.len() as u64).unwrap();
        let mut frame = [0.0, 2.0];
        index.blend(&mut frame, 0.0).unwrap();
        assert_eq!(frame, [0.0, 2.0]);
        index.blend(&mut frame, 0.5).unwrap();
        assert_eq!(frame, [1.0, 3.0]);
    }
    #[test]
    fn ivf_selects_nearest_cluster_and_ignores_noncontiguous_ids() {
        let b = ivf();
        let index = RetrievalIndex::read(&b[..], b.len() as u64).unwrap();
        let mut frames = [0.0, 0.0, 10.0, 10.0];
        index.blend(&mut frames, 1.0).unwrap();
        assert_eq!(frames, [1.0, 1.0, 11.0, 11.0]);
    }
    #[test]
    fn rejects_truncated_and_unsupported_files() {
        let b = ivf();
        for n in [0, 4, 40, b.len() - 1] {
            assert!(RetrievalIndex::read(&b[..n], n as u64).is_err());
        }
        let mut b = ivf();
        b[0] = b'X';
        assert!(RetrievalIndex::read(&b[..], b.len() as u64).is_err());
    }
}
