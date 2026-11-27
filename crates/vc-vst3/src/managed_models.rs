//! Local model discovery and conversion, worker thread only.
use std::{path::{Path,PathBuf},sync::atomic::{AtomicBool,Ordering}};
use crate::{config::PluginConfig,diagnostics::Diagnostics};
const SLOTS:[&str;4]=["toddler","girl","male","robot"];
fn files(dir:&Path, ext:&str)->Vec<PathBuf>{
    let mut paths:Vec<_>=std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e|e.path()).filter(|p|p.is_file() && p.extension().is_some_and(|x|x.eq_ignore_ascii_case(ext))).collect();
    paths.sort();paths
}
fn common(dir:&Path, names:&[&str], preferred:&str)->Option<PathBuf>{
    let exact=dir.join(preferred);
    if exact.is_file(){return Some(exact);}
    let found:Vec<_>=files(dir,"onnx").into_iter().filter(|p|p.file_stem().is_some_and(|s|{let s=s.to_string_lossy().to_lowercase();names.iter().any(|n|s.contains(n))})).collect();
    if found.len()==1 {found.into_iter().next()}else{None}
}
fn find_shared(root:&Path, current:&Path, names:&[&str], preferred:&str)->Option<PathBuf>{
    for folder in [root.join("models/common"),root.join("models"),root.to_path_buf()] {
        if let Some(path)=common(&folder,names,preferred){return Some(path);}
    }
    current.is_file().then(||current.to_path_buf())
}
pub(crate) fn missing_shared_message(config:&PluginConfig,root:&Path)->String{
    let mut missing=Vec::new();
    if !config.embedder.is_file() && !config.voice_slots.iter().any(|s|s.model.is_file()&&s.embedder.is_file()){
        missing.push("ContentVec (contentvec-f.onnx)");
    }
    if !config.f0_model.is_file() && !config.voice_slots.iter().any(|s|s.model.is_file()&&s.f0_model.is_file()){
        missing.push("RMVPE (rmvpe_20231006.onnx)");
    }
    if missing.is_empty(){format!("No complete model set in {}",root.join("models").display())}
    else{format!("Missing {}. Place ONNX files in {}",missing.join(" + "),root.join("models/common").display())}
}

pub(crate) fn discover(root:&Path, config:&mut PluginConfig, running:&AtomicBool, log:&Diagnostics){
    // The local profile is authoritative for per-character settings, even if
    // Wave Link restores an older host preset. Do not overwrite paths or keys.
    if let Ok(saved) = crate::diagnostics::load_profile() {
        config.voice_profiles = saved.voice_profiles;
        log.event("VOICE_PROFILES_RESTORED", "4 independent local profiles");
    }
    let models=root.join("models");
    let shared=models.join("common");
    log.event("MODEL_SCAN",&format!("root={} common={} onnx_files={:?}",root.display(),shared.display(),files(&shared,"onnx")));
    if let Some(p)=find_shared(root,&config.embedder,&["contentvec","hubert"],"contentvec-f.onnx") {
        if config.embedder!=p{config.embedder=p;config.embedder_output=None;config.embedder_package=None;}
        log.event("COMMON_MODEL_REGISTERED",&format!("ContentVec {}",config.embedder.display()));
    }else{log.event("COMMON_MODEL_MISSING",&format!("ContentVec: expected contentvec-f.onnx in {}; previous={}",shared.display(),config.embedder.display()));config.embedder.clear();}
    if let Some(p)=find_shared(root,&config.f0_model,&["rmvpe"],"rmvpe_20231006.onnx") {
        config.f0_model=p;
        log.event("COMMON_MODEL_REGISTERED",&format!("RMVPE {}",config.f0_model.display()));
    }else{log.event("COMMON_MODEL_MISSING",&format!("RMVPE: expected rmvpe_20231006.onnx in {}; previous={}",shared.display(),config.f0_model.display()));config.f0_model.clear();}
    if config.chunk_ms==500 && config.crossfade_ms==85 && config.sola_search_ms==12 && config.rvc_output_tail_discard_ms==10 && config.extra_convert_ms==100 {
        config.chunk_ms=200;
        config.crossfade_ms=40;
        log.event("LATENCY_PROFILE","legacy default 500/85 ms -> 200/40 ms");
    }
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
        assert_eq!(c.voice_slots[0].model,models.join("toddler/toddler.onnx"));assert_eq!(c.voice_slots[1].model,models.join("girl/custom.onnx"));assert_eq!(c.voice_slots[2].model,PathBuf::from("manual.onnx"));assert_eq!(c.embedder,models.join("common/contentvec.onnx"));assert_eq!(c.f0_model,models.join("common/rmvpe.onnx"));assert!(c.has_models());
        std::fs::remove_file(models.join("common/rmvpe.onnx")).unwrap();
        discover(&root,&mut c,&AtomicBool::new(true),&Diagnostics::default());
        assert!(c.f0_model.as_os_str().is_empty());
        assert!(!c.has_models());
        assert!(missing_shared_message(&c,&root).contains("RMVPE"));
        let _=std::fs::remove_dir_all(root);
    }
}
