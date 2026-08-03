//! Standalone diagnostic command exercising [`crate::commands::remote::http`] end to end inside
//! the real Cockle/JupyterLite Web Worker. It isolates browser transport and CORS failures from
//! repository, authentication, and database behavior in `clone`/`pull`.
//!
//! Deliberately outside `remote::client` and does not touch a Gen repository or database, so it
//! can run as `gen __emscripten-http-test <url> [...]` against any test server from the terminal.

use core::error::Error;

use clap::Args;
use serde::Serialize;
use serde_json::to_string as json_to_string;

use crate::commands::remote::http::{self, HttpRequest};

#[derive(Args)]
pub struct Command {
    /// URL to request.
    #[clap(index = 1)]
    pub url: String,
    /// HTTP method to use.
    #[arg(short, long, default_value = "GET")]
    pub method: String,
    /// Request headers, formatted as "Name: Value". May be repeated.
    #[arg(short = 'H', long = "header")]
    pub headers: Vec<String>,
    /// Request body, sent as-is (UTF-8). Omit for no body.
    #[arg(short, long)]
    pub body: Option<String>,
}

#[derive(Serialize)]
struct FetchReport {
    ok: bool,
    status: Option<u16>,
    body_length: Option<usize>,
    body_preview: Option<String>,
    error: Option<String>,
}

fn parse_header(raw: &str) -> Result<(&str, &str), String> {
    let (name, value) = raw
        .split_once(':')
        .ok_or_else(|| format!("header {raw:?} is not in \"Name: Value\" form"))?;
    Ok((name.trim(), value.trim()))
}

pub fn execute(command: &Command) -> Result<(), Box<dyn Error>> {
    let headers = command
        .headers
        .iter()
        .map(|raw| parse_header(raw))
        .collect::<Result<Vec<_>, _>>()?;
    let body = command.body.as_deref().map(str::as_bytes);

    let report = match http::request(HttpRequest {
        method: &command.method,
        url: &command.url,
        headers: &headers,
        body,
    }) {
        Ok(response) => FetchReport {
            ok: true,
            status: Some(response.status),
            body_length: Some(response.body.len()),
            body_preview: Some(
                String::from_utf8_lossy(&response.body[..response.body.len().min(200)])
                    .into_owned(),
            ),
            error: None,
        },
        Err(error) => FetchReport {
            ok: false,
            status: None,
            body_length: None,
            body_preview: None,
            error: Some(error.to_string()),
        },
    };

    println!("{}", json_to_string(&report)?);
    Ok(())
}
