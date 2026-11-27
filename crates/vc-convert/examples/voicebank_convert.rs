//! Local-only VoiceBank conversion helper. No Python, network, or inference.
use anyhow::{bail, Context, Result};
use std::{io::Write, path::PathBuf};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(
        args.next()
            .context("usage: voicebank_convert input.pth output.onnx")?,
    );
    let output = PathBuf::from(args.next().context("missing output.onnx")?);
    if args.next().is_some() {
        bail!("unexpected argument");
    }
    if output.exists() {
        bail!("output already exists: {}", output.display());
    }
    println!("Input: {}", input.display());
    let bytes = std::fs::read(&input).context("read PTH")?;
    let conversion = vc_convert::pth_to_onnx(
        &bytes,
        &vc_convert::ConvertOptions::default(),
        &mut |stage| println!("{}", stage.label()),
    )?;
    let tmp = output.with_extension("onnx.tmp");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .context("create temporary ONNX")?;
    let result = file
        .write_all(&conversion.onnx_bytes)
        .and_then(|()| file.sync_all());
    drop(file);
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).context("write ONNX");
    }
    if let Err(e) = std::fs::rename(&tmp, &output) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).context("publish ONNX");
    }
    println!(
        "RVC={} sample_rate={} bytes={}",
        conversion.version.as_str(),
        conversion.sample_rate,
        conversion.onnx_bytes.len()
    );
    println!("OK: {}", output.display());
    Ok(())
}
