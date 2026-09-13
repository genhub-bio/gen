#[cfg(not(target_os = "emscripten"))]
use std::{
    io::prelude::*,
    net::TcpListener,
    sync::mpsc,
    thread::{self, JoinHandle},
};

#[cfg(not(target_os = "emscripten"))]
use rand::{rng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
#[cfg(not(target_os = "emscripten"))]
use url::Url;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthTokens {
    pub jwt: String,
    pub refresh_token: String,
}

/// Starts the localhost callback listener used by the native login flow. Unavailable on
/// Emscripten, which has no real TCP sockets; the browser login flow
/// (`commands::remote::browser`) receives the callback through a browser-hosted page and a
/// JS bridge instead.
#[cfg(not(target_os = "emscripten"))]
#[allow(clippy::type_complexity)]
pub fn start_callback_server(
    expected_state: String,
) -> Result<(String, JoinHandle<()>, mpsc::Receiver<AuthTokens>), Box<dyn std::error::Error>> {
    // Pick random port
    let mut ports: Vec<u16> = (40000..=50000).collect();
    ports.shuffle(&mut rng());

    let mut listener_result = None;
    for (attempt, &port) in ports.iter().take(10).enumerate() {
        match TcpListener::bind(("127.0.0.1", port)) {
            Ok(listener) => {
                listener_result = Some(listener);
                break;
            }
            Err(_) if attempt == 9 => {
                return Err("Failed to bind to any port after 10 attempts".into());
            }
            _ => {}
        }
    }

    let listener = listener_result.ok_or("No available port found")?;
    let local_addr = listener.local_addr()?.to_string();
    let state_clone = expected_state.clone();

    let (tx, rx) = mpsc::channel::<AuthTokens>();

    // Spawn thread to handle callback
    let handle = thread::spawn(move || {
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    let mut buffer = [0; 1024];
                    if stream.read(&mut buffer).is_err() {
                        eprintln!("Failed to read stream");
                        continue;
                    }

                    let response = "HTTP/1.1 200 OK\r\n\r\nCallback received. Please check command line output for additional information. You can close this window.";
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();

                    let mut headers = [httparse::EMPTY_HEADER; 64];
                    let mut req = httparse::Request::new(&mut headers);
                    if req.parse(&buffer).is_err() {
                        eprintln!("Error parsing request");
                        break;
                    }
                    let res_path = req.path.unwrap_or_default();

                    let base = Url::parse("http://localhost").unwrap();
                    let url = base.join(res_path).unwrap();

                    let mut returned_state: Option<String> = None;
                    let mut jwt: Option<String> = None;
                    let mut refresh_token: Option<String> = None;

                    for (k, v) in url.query_pairs() {
                        match k.as_ref() {
                            "state" => returned_state = Some(v.into_owned()),
                            "jwt" => jwt = Some(v.into_owned()),
                            "refresh_token" => refresh_token = Some(v.into_owned()),
                            _ => {}
                        }
                    }

                    if let (Some(returned_state), Some(jwt), Some(refresh_token)) =
                        (returned_state, jwt, refresh_token)
                    {
                        if returned_state != state_clone {
                            eprintln!("State mismatch!");
                            break;
                        }

                        // Send tokens back
                        let _ = tx.send(AuthTokens { jwt, refresh_token });
                        return; // stop after success
                    } else {
                        eprintln!("Invalid callback, missing params");
                    }

                    break;
                }
                Err(_) => {
                    eprintln!("Failed to establish connection");
                    break;
                }
            }
        }
    });

    Ok((local_addr, handle, rx))
}
