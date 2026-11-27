//! Loopback HTTP control, independent from audio processing. One active instance
//! owns the fixed port; never bind to a LAN address. No CORS or external assets.
use crate::{hotkeys::{Binding, HotkeyConfig, supported_keys, key_name}, params::VcRvcParams, runtime::ReloadWaker};
use std::{io::{self, Read, Write}, net::{TcpListener, TcpStream}, sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}}, thread::{self, JoinHandle}, time::{Duration, Instant}};
const HOST: &str = "127.0.0.1:18765";
const ORIGIN: &str = "http://127.0.0.1:18765";
const HTML: &str = include_str!("control.html");
const MAX: usize = 8192;

type Target=(Arc<VcRvcParams>,Arc<ReloadWaker>);
struct SharedHttp { clients:Arc<Mutex<Vec<Target>>>, stop:Arc<AtomicBool>, worker:JoinHandle<()> }
static SERVER:Mutex<Option<SharedHttp>>=Mutex::new(None);
pub(crate) struct HttpControl { params:Arc<VcRvcParams> }
impl HttpControl {
    pub(crate) fn start(params: Arc<VcRvcParams>, waker: Arc<ReloadWaker>) -> io::Result<Self> {
        let mut server=SERVER.lock().unwrap_or_else(|e|e.into_inner());
        if let Some(shared)=server.as_ref(){
            shared.clients.lock().unwrap_or_else(|e|e.into_inner()).push((params.clone(),waker));
            params.diagnostics.event("HTTP_JOIN","shared localhost:18765");
            return Ok(Self{params});
        }
        let listener = TcpListener::bind(HOST)?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let clients=Arc::new(Mutex::new(vec![(params.clone(),waker)]));
        let targets=clients.clone();
        let diag=params.diagnostics.clone();
        let worker = thread::Builder::new().name("voicebank-http".into()).spawn(move || {
            diag.event("HTTP_START", ORIGIN);
            while !flag.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        let snapshot=targets.lock().unwrap_or_else(|e|e.into_inner()).clone();
                        if let Some((p,w))=snapshot.iter().max_by_key(|(p,_)|(p.loaded_voice_mask.load(Ordering::Acquire).count_ones(),p.audio_seen.load(Ordering::Acquire))){
                            let result = handle(&mut stream,p,w,&snapshot);
                            if let Err(err) = result { p.diagnostics.event("HTTP_ERROR", &err.to_string()); }
                        }
                    }
                    Err(err) if err.kind() == io::ErrorKind::WouldBlock => thread::park_timeout(Duration::from_millis(10)),
                    Err(err) => { diag.event("HTTP_ERROR", &err.to_string()); break; }
                }
            }
            diag.event("HTTP_STOP", ORIGIN);
        })?;
        *server=Some(SharedHttp{clients,stop,worker});
        Ok(Self { params })
    }
}
impl Drop for HttpControl {
    fn drop(&mut self) {
        let mut server=SERVER.lock().unwrap_or_else(|e|e.into_inner());
        let empty=if let Some(shared)=server.as_ref(){
            let mut clients=shared.clients.lock().unwrap_or_else(|e|e.into_inner());
            clients.retain(|(p,_)|!Arc::ptr_eq(p,&self.params));
            clients.is_empty()
        }else{false};
        if empty {if let Some(shared)=server.take(){shared.stop.store(true,Ordering::Release);shared.worker.thread().unpark();let _=shared.worker.join();}}
    }
}

fn voice(path: &str) -> Option<i32> {
    Some(match path {
        "/api/voice/original" => 0, "/api/voice/toddler" => 1, "/api/voice/girl" => 2,
        "/api/voice/male" => 3, "/api/voice/robot" => 4, _ => return None,
    })
}
fn decode_shortcuts(body: &str) -> Result<HotkeyConfig, String> {
    let mut fields = body.split(';');
    let enabled = match fields.next() { Some("1") => true, Some("0") => false, _ => return Err("Invalid enabled flag".into()) };
    let mut bindings = [Binding::default(); 6];
    for binding in &mut bindings {
        let (key, modifiers) = fields.next().and_then(|s| s.split_once(':')).ok_or("Missing binding")?;
        *binding = Binding { key: key.parse().map_err(|_| "Invalid key")?, modifiers: modifiers.parse().map_err(|_| "Invalid modifiers")? };
    }
    if fields.next().is_some() { return Err("Too many bindings".into()); }
    let config = HotkeyConfig { enabled, bindings };
    config.validate()?;
    Ok(config)
}
fn state(params: &VcRvcParams) -> String {
    let settings = params.settings.read().unwrap_or_else(|e|e.into_inner());
    let diag=&params.diagnostics;
    let rate=diag.host_sample_rate.load(Ordering::Relaxed);
    let chunks=diag.chunks.load(Ordering::Relaxed);
    let switches=diag.switch_count.load(Ordering::Acquire);
    serde_json::json!({
        "selected":params.selected_voice(),
        "active":params.active_voice.load(Ordering::Acquire),
        "ready_mask":params.loaded_voice_mask.load(Ordering::Acquire),
        "status":params.model_message.lock().unwrap_or_else(|e|e.into_inner()).clone(),
        "enabled":settings.hotkeys.enabled,
        "bindings":settings.hotkeys.bindings.iter().map(|b|[b.key as u32,b.modifiers as u32]).collect::<Vec<_>>(),
        "diagnostics":{
            "host_latency_samples":diag.host_latency_samples.load(Ordering::Relaxed),
            "host_sample_rate":rate,
            "processing_ms":(chunks>0).then(||diag.last_processing_us.load(Ordering::Relaxed) as f64/1000.0),
            "switch_enqueue_ms":(switches>0).then(||diag.last_switch_us.load(Ordering::Relaxed) as f64/1000.0),
            "chunks":chunks,
            "input_dropped":diag.input_dropped.load(Ordering::Relaxed),
            "output_missing":diag.output_missing.load(Ordering::Relaxed),
            "output_dropped":diag.output_dropped.load(Ordering::Relaxed),
            "error":diag.runtime_error.lock().unwrap_or_else(|e|e.into_inner()).clone(),
            "profile_error":diag.profile_error(),
            "log_error":diag.error(),
            "log_path":diag.path()
        }
    }).to_string()
}
fn profile_slot(path:&str)->Option<usize>{
    match path.strip_prefix("/api/profile/")? {"toddler"=>Some(0),"girl"=>Some(1),"male"=>Some(2),"robot"=>Some(3),_=>None}
}
fn profiles(params:&VcRvcParams)->String {
    let settings=params.settings.read().unwrap_or_else(|e|e.into_inner());
    serde_json::json!({"saved":settings.voice_profiles,"defaults":std::array::from_fn::<_,4,_>(crate::config::VoiceProfile::initial)}).to_string()
}
fn decode_profile(body:&str,slot:usize,settings:&crate::config::PluginConfig)->anyhow::Result<crate::config::VoiceProfile> {
    use crate::config::VoiceProfile;
    let value:serde_json::Value=serde_json::from_str(body)?;
    let expected=serde_json::to_value(VoiceProfile::initial(slot))?;
    let actual=value.as_object().ok_or_else(||anyhow::anyhow!("Profile must be a JSON object"))?;
    let fields=expected.as_object().expect("profile object");
    anyhow::ensure!(actual.len()==fields.len() && fields.keys().all(|k|actual.contains_key(k)),"Profile fields missing or unknown");
    let profile:VoiceProfile=serde_json::from_value(value)?;
    profile.validate_live()?;
    let mut candidate=settings.clone();
    candidate.voice_profiles[slot]=profile.clone();
    candidate.for_voice(slot).validate()?;
    Ok(profile)
}
fn save_voice_profile(params:&VcRvcParams,waker:&ReloadWaker,targets:&[Target],slot:usize,body:&str)->Result<(),( &'static str,String)> {
    // Validate and persist before publishing changes to the live settings.
    // Only this slot is modified; shortcuts, model paths and other slots stay intact.
    let mut settings=params.settings.write().unwrap_or_else(|e|e.into_inner());
    let profile=decode_profile(body,slot,&settings).map_err(|e|("400 Bad Request",e.to_string()))?;
    let mut next=settings.clone();
    next.voice_profiles[slot]=profile.clone();
    next.startup_voice=params.selected_voice();
    if !cfg!(test) {
        crate::diagnostics::save_profile(&next).map_err(|e|{
            params.diagnostics.set_profile_error(Some(e.to_string()));
            ("500 Internal Server Error",e.to_string())
        })?;
    }
    *settings=next;
    let reload=settings.has_voice_bank();
    drop(settings);
    params.diagnostics.set_profile_error(None);
    if reload {params.profile_reload_requested.store(true,Ordering::Release);waker.wake();}
    for (p,w) in targets {
        if !std::ptr::eq(p.as_ref(),params) {
            let mut config=p.settings.write().unwrap_or_else(|e|e.into_inner());
            config.voice_profiles[slot]=profile.clone();
            let reload=config.has_voice_bank() && (p.audio_seen.load(Ordering::Acquire)||p.loaded_voice_mask.load(Ordering::Acquire)!=0);
            drop(config);
            if reload {p.profile_reload_requested.store(true,Ordering::Release);w.wake();}
        }
    }
    params.diagnostics.event("HTTP_PROFILE",&format!("slot={} saved; reload_requested={reload}",slot+1));
    Ok(())
}
fn keys() -> String {
    // Labels generated by key_name are fixed ASCII without quotes.
    format!("[{}]", supported_keys().map(|key| format!("[{},\"{}\"]",key,key_name(key))).collect::<Vec<_>>().join(","))
}
fn reply(stream: &mut TcpStream, status: &str, mime: &str, body: &str) -> io::Result<()> {
    write!(stream,"HTTP/1.1 {status}\r\nContent-Type: {mime}; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nContent-Security-Policy: default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'\r\n\r\n{body}",body.len())?;
    stream.flush()
}
fn allowed_request(method: &str, path: &str, host: Option<&str>, origin: Option<&str>, site: Option<&str>) -> bool {
    let expected_origin = match host {
        Some(HOST) => ORIGIN,
        Some("localhost:18765") => "http://localhost:18765",
        _ => return false,
    };
    // Opening the read-only HTML page from a local file or another site is
    // harmless. Its subsequent API requests must still be same-origin.
    if method == "GET" && matches!(path, "/" | "/control.html") {
        return true;
    }
    origin.is_none_or(|value| value == expected_origin)
        && site.is_none_or(|value| value == "same-origin" || value == "none")
}
fn handle(stream: &mut TcpStream, params: &VcRvcParams, waker: &ReloadWaker, targets:&[Target]) -> io::Result<()> {
    // Accepted sockets inherit nonblocking mode on some Windows hosts.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_millis(200)))?;
    stream.set_write_timeout(Some(Duration::from_millis(200)))?;
    let deadline = Instant::now() + Duration::from_millis(800);
    let mut data = Vec::new();
    let mut buf = [0u8; 1024];
    let head_end = loop {
        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") { break i+4; }
        if data.len() >= MAX || Instant::now() >= deadline { return reply(stream,"400 Bad Request","text/plain","Invalid request"); }
        let n = stream.read(&mut buf)?;
        if n == 0 { return Ok(()); }
        data.extend_from_slice(&buf[..n]);
    };
    let header = match std::str::from_utf8(&data[..head_end]) { Ok(s)=>s, Err(_)=>return reply(stream,"400 Bad Request","text/plain","Invalid header") };
    let mut lines = header.lines();
    let mut request = lines.next().unwrap_or("").split_whitespace();
    let method = request.next().unwrap_or("").to_owned();
    let path = request.next().unwrap_or("").to_owned();
    let mut host = None;
    let mut origin = None;
    let mut site = None;
    let mut len = 0usize;
    for line in lines {
        if let Some((k,v)) = line.split_once(':') {
            let v = v.trim();
            match k.to_ascii_lowercase().as_str() {
                "host" => host=Some(v),
                "origin" => origin=Some(v),
                "sec-fetch-site" => site=Some(v),
                "content-length" => len=v.parse().unwrap_or(MAX+1),
                "transfer-encoding" => return reply(stream,"400 Bad Request","text/plain","Unsupported transfer encoding"),
                _=>{}
            }
        }
    }
    // Restrict API calls to loopback same-origin or native local HTTP clients.
    if !allowed_request(&method, &path, host, origin, site) {
        params.diagnostics.event("HTTP_REJECT", &format!("method={method} path={path} host={host:?} origin={origin:?} site={site:?}"));
        return reply(stream,"403 Forbidden","text/plain","API request rejected: use the control page at http://127.0.0.1:18765/control.html or a local native HTTP client.");
    }
    if len>MAX || head_end+len>MAX { return reply(stream,"413 Payload Too Large","text/plain","Request too large"); }
    while data.len()<head_end+len {
        if Instant::now()>=deadline { return reply(stream,"408 Request Timeout","text/plain","Timeout"); }
        let n=stream.read(&mut buf)?;
        if n==0 { return Ok(()); }
        data.extend_from_slice(&buf[..n]);
    }
    if method=="GET" {
        return match path.as_str() {
            "/"|"/control.html"=>reply(stream,"200 OK","text/html",HTML),
            "/api/state"=>reply(stream,"200 OK","application/json",&state(params)),
            "/api/keys"=>reply(stream,"200 OK","application/json",&keys()),
            "/api/profiles"=>reply(stream,"200 OK","application/json",&profiles(params)),
            _=>reply(stream,"404 Not Found","text/plain","Not found"),
        };
    }
    if method!="POST" { return reply(stream,"405 Method Not Allowed","text/plain","Use POST"); }
    if let Some(slot)=voice(&path) {
        params.set_selected_voice(slot);
        waker.wake();
        for (p,w) in targets {if !std::ptr::eq(p.as_ref(),params){p.set_selected_voice(slot);w.wake();}}
        if !params.voice_loaded(slot){params.diagnostics.event("HTTP_VOICE_UNAVAILABLE",&format!("slot={slot}: model not ready"));}
        params.diagnostics.event("HTTP_VOICE",&format!("Requested voice slot {slot}"));
        return reply(stream,"202 Accepted","application/json",&state(params));
    }
    if let Some(slot)=profile_slot(&path) {
        let body=match std::str::from_utf8(&data[head_end..head_end+len]) {Ok(s)=>s,Err(_)=>return reply(stream,"400 Bad Request","text/plain","Invalid UTF-8")};
        return match save_voice_profile(params,waker,targets,slot,body) {
            Ok(())=>reply(stream,"200 OK","application/json",&profiles(params)),
            Err((status,err))=>reply(stream,status,"text/plain",&err),
        };
    }
    if path=="/api/shortcuts" {
        let parsed=std::str::from_utf8(&data[head_end..head_end+len]).map_err(|_|"Invalid UTF-8".to_owned()).and_then(decode_shortcuts);
        return match parsed {
            Ok(config)=>{
                params.settings.write().unwrap_or_else(|e|e.into_inner()).hotkeys=config;
                for (p,_) in targets {if !std::ptr::eq(p.as_ref(),params){p.settings.write().unwrap_or_else(|e|e.into_inner()).hotkeys=config;}}
                params.request_save();
                params.flush_profile();
                let error=params.diagnostics.profile_error();
                params.diagnostics.event("HTTP_SHORTCUTS",if error.is_some() {"Changed in memory; disk save failed"} else {"Updated"});
                if let Some(error)=error {reply(stream,"500 Internal Server Error","text/plain",&error)}
                else {reply(stream,"200 OK","application/json",&state(params))}
            }
            Err(err)=>reply(stream,"400 Bad Request","text/plain",&err),
        };
    }
    reply(stream,"404 Not Found","text/plain","Not found")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn local_html_navigation_and_api_origin_rules() {
        for site in [Some("cross-site"), Some("none"), Some("same-origin"), None] {
            assert!(allowed_request("GET", "/control.html", Some(HOST), None, site));
        }
        assert!(allowed_request("GET", "/", Some("localhost:18765"), None, Some("cross-site")));
        assert!(allowed_request("POST", "/api/voice/girl", Some("localhost:18765"), Some("http://localhost:18765"), Some("same-origin")));
        assert!(allowed_request("POST", "/api/voice/girl", Some(HOST), None, None));
        assert!(!allowed_request("POST", "/api/voice/girl", Some(HOST), Some("null"), Some("cross-site")));
        assert!(!allowed_request("POST", "/api/voice/girl", Some(HOST), Some("https://example.com"), None));
        assert!(!allowed_request("GET", "/api/state", Some(HOST), None, Some("cross-site")));
        assert!(!allowed_request("GET", "/control.html", Some("example.com:18765"), None, None));
    }
    #[test] fn profile_api_changes_only_one_slot_and_rejects_invalid_payloads() {
        let p=VcRvcParams::default();
        let w=ReloadWaker::default();
        let before=p.settings.read().unwrap().clone();
        let mut profile=before.voice_profiles[1].clone();
        profile.pitch_shift=9.0;
        profile.input_f0_min_hz=70.0;
        profile.formant_shift=1.0;
        let body=serde_json::to_string(&profile).unwrap();
        save_voice_profile(&p,&w,&[],1,&body).unwrap();
        let after=p.settings.read().unwrap().clone();
        assert_eq!(after.voice_profiles[1],profile);
        for i in [0,2,3] {assert_eq!(after.voice_profiles[i],before.voice_profiles[i]);}
        assert_eq!(after.hotkeys,before.hotkeys);
        assert!(save_voice_profile(&p,&w,&[],1,"{}").is_err());
        profile.chunk_ms=19;
        assert!(save_voice_profile(&p,&w,&[],1,&serde_json::to_string(&profile).unwrap()).is_err());
        assert_eq!(*p.settings.read().unwrap(),after);
        let response:serde_json::Value=serde_json::from_str(&profiles(&p)).unwrap();
        assert_eq!(response["saved"][1]["pitch_shift"],9.0);
        assert_eq!(response["defaults"][1]["pitch_shift"],12.0);
        assert!(serde_json::from_str::<serde_json::Value>(&state(&p)).unwrap()["diagnostics"]["processing_ms"].is_null());
    }
    #[test] fn all_voice_routes() {
        for (i,name) in ["original","toddler","girl","male","robot"].iter().enumerate() {
            assert_eq!(voice(&format!("/api/voice/{name}")),Some(i as i32));
        }
        assert_eq!(voice("/api/voice/unknown"),None);
    }
    #[test] fn shortcuts_validate_duplicates_and_ranges() {
        let c=decode_shortcuts("1;118:0;119:0;120:0;121:0;122:0;123:0").unwrap();
        assert_eq!(c,HotkeyConfig::default());
        assert!(decode_shortcuts("1;118:0;118:0;120:0;121:0;122:0;123:0").is_err());
        assert!(decode_shortcuts("1;118:16;119:0;120:0;121:0;122:0;123:0").is_err());
    }
    #[test] fn local_http_request_updates_voice_and_rejects_foreign_origin() {
        let params=VcRvcParams::default();
        let waker=ReloadWaker::default();
        let listener=TcpListener::bind("127.0.0.1:0").unwrap();
        let addr=listener.local_addr().unwrap();
        for (extra,expected) in [("Origin: https://example.com\r\n", "403 Forbidden"),("","202 Accepted")] {
            let request=format!("POST /api/voice/robot HTTP/1.1\r\nHost: {HOST}\r\n{extra}Content-Length: 0\r\n\r\n");
            let client=thread::spawn(move || {
                let mut s=TcpStream::connect(addr).unwrap();
                s.write_all(request.as_bytes()).unwrap();
                let mut response=String::new();
                s.read_to_string(&mut response).unwrap();
                response
            });
            let (mut stream,_)=listener.accept().unwrap();
            handle(&mut stream,&params,&waker,&[]).unwrap();
            drop(stream);
            assert!(client.join().unwrap().contains(expected));
        }
        assert_eq!(params.selected_voice(),4);
    }
}
