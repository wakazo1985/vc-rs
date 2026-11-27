//! Local model discovery and conversion, worker thread only.
use std::{path::{Path,PathBuf},sync::atomic::{AtomicBool,Ordering}};
use crate::{config::PluginConfig,diagnostics::Diagnostics};
const SLOTS:[&str;4]=["toddler","girl","male","robot"];
fn files(dir:&Path, ext:&str)->Vec<PathBuf>{
    let mut paths:Vec<_>=std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e|e.path()).filter(|p|p.is_file() && p.extension().is_some_and(|x|x.eq_ignore_ascii_case(ext))).collect();
    paths.sort();paths
}
fn common(dir:&Path, names:&[&str])->Option<PathBuf>{
    let found:Vec<_>=files(dir,"onnx").into_iter().filter(|p|p.file_stem().is_some_and(|s|{let s=s.to_string_lossy().to_lowercase();names.iter().any(|n|s.contains(n))})).collect();
    if found.len()==1 {found.into_iter().next()}else{None}
}
pub(crate) fn discover(root:&Path, config:&mut PluginConfig, running:&AtomicBool, log:&Diagnostics){
    let models=root.join("models");
    let shared=models.join("common");
    if let Some(p)=common(&shared,&["contentvec","hubert"]) {config.embedder=p;config.embedder_output=None;config.embedder_package=None;}
    if let Some(p)=common(&shared,&["rmvpe"]) {config.f0_model=p;}
    for (i,name) in SLOTS.iter().enumerate(){
        if !running.load(Ordering::Acquire){return;}
        let dir=models.join(name);
        let dest=dir.join(format!("{name}.onnx"));
        let pths=files(&dir,"pth");
        let indices=files(&dir,"index");
        if indices.len()==1 {config.voice_slots[i].index=indices[0].clone();log.event("INDEX_REGISTERED",name);}
        else if indices.len()>1 {log.event("INDEX_ERROR",&format!("{name}: multiple INDEX files; keep one"));}
        if !dest.is_file() && pths.len()==1 {
            #[cfg(windows)]
            {
                let exe=root.join("tools").join("voicebank_convert.exe");
                if exe.is_file(){
                    use std::{os::windows::process::CommandExt,process::{Command,Stdio},time::Duration};
                    let logs=root.join("logs");
                    let result=(||->std::io::Result<bool>{
                        std::fs::create_dir_all(&logs)?;
                        let mut child=Command::new(exe).arg(&pths[0]).arg(&dest).creation_flags(0x08000000)
                            .stdout(Stdio::from(std::fs::File::create(logs.join(format!("{name}-conversion.stdout.log")))?))
                            .stderr(Stdio::from(std::fs::File::create(logs.join(format!("{name}-conversion.stderr.log")))?)).spawn()?;
                        loop{
                            if let Some(status)=child.try_wait()? {return Ok(status.success());}
                            if !running.load(Ordering::Acquire){let _=child.kill();let _=child.wait();return Ok(false);}
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    })();
                    log.event("MODEL_CONVERSION",&format!("{name}: {result:?}"));
                }else{log.event("MODEL_ERROR",&format!("{name}: converter missing"));}
            }
        }else if !dest.is_file() && pths.len()>1 {log.event("MODEL_ERROR",&format!("{name}: multiple PTH files; keep one checkpoint"));}
        if dest.is_file(){config.voice_slots[i].model=dest;log.event("MODEL_REGISTERED",name);}
        else {let onnx=files(&dir,"onnx");if onnx.len()==1 {config.voice_slots[i].model=onnx[0].clone();log.event("MODEL_REGISTERED",name);}else{log.event("MODEL_MISSING",name);}}
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]fn discovers_named_models_without_replacing_missing_manual_paths(){
        let root=std::env::temp_dir().join(format!("voicebank-managed-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let models=root.join("models");for dir in ["toddler","girl","common"]{std::fs::create_dir_all(models.join(dir)).unwrap();}
        for p in ["toddler/toddler.onnx","girl/custom.onnx","common/contentvec.onnx","common/rmvpe.onnx"]{std::fs::write(models.join(p),b"fixture").unwrap();}
        let mut c=PluginConfig::default();c.voice_slots[2].model="manual.onnx".into();
        discover(&root,&mut c,&AtomicBool::new(true),&Diagnostics::default());
        assert_eq!(c.voice_slots[0].model,models.join("toddler/toddler.onnx"));assert_eq!(c.voice_slots[1].model,models.join("girl/custom.onnx"));assert_eq!(c.voice_slots[2].model,PathBuf::from("manual.onnx"));assert_eq!(c.embedder,models.join("common/contentvec.onnx"));assert_eq!(c.f0_model,models.join("common/rmvpe.onnx"));let _=std::fs::remove_dir_all(root);
    }
}
