use clap::Subcommand;
use gen_models::{
    db::ConfigConnection,
    operations::{Defaults, Remote},
};
use reqwest::{blocking::Client, redirect::Policy};
use serde::Deserialize;
use thiserror::Error;

pub mod server;
pub mod utils;

#[derive(Subcommand)]
pub enum RemoteCommand {
    /// Add a new remote repository
    Add {
        /// The name of the remote
        name: String,
        /// The URL of the remote repository
        url: String,
    },
    /// List all configured remotes
    List,
    /// Remove a remote repository
    Remove {
        /// The name of the remote to remove
        name: String,
    },
    /// Set the default remote
    SetDefault {
        /// The name of the remote to set as default
        name: String,
    },
    /// Get the current default remote
    GetDefault,
    /// Login to a remote
    Login {
        /// The remote to login. Uses default if not specified
        name: Option<String>,
    },
}

#[derive(Debug, Error, PartialEq)]
pub enum RemoteError {
    #[error("No redirect url returned for url {0}")]
    NoRedirectUrl(String),
}

#[derive(Debug, Deserialize)]
struct RemoteDiscoveryResponse {
    remote: DiscoveredRemote,
}

#[derive(Debug, Deserialize)]
struct DiscoveredRemote {
    url: String,
}

fn discovery_endpoint(remote_url: &str) -> Option<String> {
    let mut parsed = url::Url::parse(remote_url).ok()?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return None;
    }

    let segments = parsed.path_segments()?.collect::<Vec<_>>();
    if segments.is_empty() {
        return None;
    }
    if segments.last() == Some(&"remote") {
        return Some(remote_url.to_string());
    }
    if segments.contains(&"dolt") {
        return None;
    }

    let repos_index = segments.iter().position(|segment| *segment == "repos")?;
    if segments.len() < repos_index + 3 {
        return None;
    }

    let discovery_path = format!("/{}/remote", segments[..repos_index + 3].join("/"));
    parsed.set_path(&discovery_path);
    parsed.set_query(None);
    parsed.set_fragment(None);
    Some(parsed.to_string())
}

pub fn discover_dolt_remote_url(
    remote_url: &str,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    let Some(discovery_url) = discovery_endpoint(remote_url) else {
        return Ok(None);
    };

    let client = Client::builder().redirect(Policy::none()).build()?;
    let response = client.get(discovery_url).send()?.error_for_status()?;
    let discovery: RemoteDiscoveryResponse = response.json()?;
    Ok(Some(discovery.remote.url))
}

pub fn validate_dolt_remote_url(remote_url: &str) -> Result<(), Box<dyn std::error::Error>> {
    if remote_url.starts_with("file://") || remote_url.starts_with("http://") {
        return Ok(());
    }

    Err(format!(
        "Dolt remote URL `{remote_url}` is not supported by the current Doltlite build. \
Supported remote schemes are `file://` and unauthenticated `http://` only."
    )
    .into())
}

pub fn remove_remote(
    conn: &ConfigConnection,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(default_remote) = Defaults::get_default_remote(conn)
        && default_remote == name
        && let Err(err) = Defaults::set_default_remote_compat(conn, None)
    {
        eprintln!("Failed to clear default remote: {err}");
        Err(Box::new(err))
    } else {
        Remote::delete(conn, name)?;
        Ok(())
    }
}

pub fn login_remote(
    conn: &ConfigConnection,
    name: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    let remote_name = name
        .map(str::to_owned)
        .or_else(|| Defaults::get_default_remote(conn))
        .ok_or("No remote specified and no default set.")?;
    let remote = Remote::get_by_name(conn, &remote_name)?;
    let remote_url = remote.url;
    let fqdn = {
        let parsed = url::Url::parse(&remote_url)?;
        match parsed.port() {
            Some(port) => format!(
                "{}://{}:{}",
                parsed.scheme(),
                parsed.host_str().unwrap_or_default(),
                port
            ),
            None => format!(
                "{}://{}",
                parsed.scheme(),
                parsed.host_str().unwrap_or_default()
            ),
        }
    };
    println!("Logging in to remote: {fqdn}");
    let state = utils::generate_state().expect("Unable to generate random nonce.");
    let (local_addr, handle, rx) =
        server::start_callback_server(state.clone()).expect("Unable to start callback server.");

    let client = Client::builder().redirect(Policy::none()).build()?;
    let res = client
        .get(format!("{fqdn}/api/auth/cli/login/"))
        .query(&[
            ("redirect_uri", &format!("{fqdn}/api/auth/cli/callback")),
            ("state", &state),
            ("redirect_to", &format!("http://{local_addr}")),
        ])
        .send()?;
    if let Some(location) = res.headers().get("location") {
        let redirect_url = location.to_str()?;
        println!("Redirecting to: {redirect_url}");

        webbrowser::open(redirect_url)?;
    } else {
        println!("No redirect URL found. Response: {res:?}");
        return Err(Box::new(RemoteError::NoRedirectUrl(remote_url)));
    }

    handle.join().unwrap();

    if let Ok(tokens) = rx.recv() {
        utils::save_tokens(&remote_name, &tokens).expect("Failed to save login information.");
    }

    Ok(())
}

/// Handle remote management commands with comprehensive error handling
pub fn handle_remote_command(
    conn: &ConfigConnection,
    command: &RemoteCommand,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        RemoteCommand::Add { name, url } => match Remote::create(conn, name, url) {
            Ok(_) => {
                println!("Remote '{name}' added successfully");
                Ok(())
            }
            Err(remote_err) => {
                eprintln!("Error: {remote_err}");
                Err(Box::new(remote_err))
            }
        },

        RemoteCommand::List => {
            let remotes = Remote::list_all(conn);
            if remotes.is_empty() {
                println!("No remotes configured");
            } else {
                println!("Configured remotes:");
                for remote in remotes {
                    println!("  {} -> {}", remote.name, remote.url);
                }
            }
            Ok(())
        }

        RemoteCommand::Remove { name } => match remove_remote(conn, name) {
            Ok(_) => {
                println!("Remote '{name}' removed successfully");
                Ok(())
            }
            Err(remote_err) => {
                eprintln!("Error: {remote_err}");
                Err(remote_err)
            }
        },

        RemoteCommand::SetDefault { name } => {
            match Defaults::set_default_remote(conn, Some(name)) {
                Ok(_) => {
                    println!("Default remote set to '{name}'");
                    Ok(())
                }
                Err(remote_err) => {
                    eprintln!("Error: {remote_err}");
                    Err(Box::new(remote_err))
                }
            }
        }

        RemoteCommand::GetDefault => match Defaults::get_default_remote(conn) {
            Some(remote_name) => {
                println!("Default remote: {remote_name}");
                Ok(())
            }
            None => {
                println!("No default remote configured");
                Ok(())
            }
        },

        RemoteCommand::Login { name } => login_remote(conn, name.as_deref()),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use super::*;
    use crate::test_helpers::setup_gen;

    #[cfg(test)]
    mod remote {
        use super::*;

        #[test]
        fn test_remote_add_command() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Test successful add
            let cmd = RemoteCommand::Add {
                name: "origin".to_string(),
                url: "https://genhub.bio/user/repo.gen".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd).is_ok());

            // Verify remote was added
            let remote = Remote::get_by_name(op_conn, "origin").unwrap();
            assert_eq!(remote.name, "origin");
            assert_eq!(remote.url, "https://genhub.bio/user/repo.gen");

            // Test duplicate name error
            let cmd_duplicate = RemoteCommand::Add {
                name: "origin".to_string(),
                url: "https://different.com/repo.gen".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_duplicate).is_err());
        }

        #[test]
        fn test_remote_add_validation_errors() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Test invalid name
            let cmd_invalid_name = RemoteCommand::Add {
                name: "invalid name".to_string(),
                url: "https://genhub.bio/user/repo.gen".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_invalid_name).is_err());

            // Test invalid URL
            let cmd_invalid_url = RemoteCommand::Add {
                name: "origin".to_string(),
                url: "not-a-url".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_invalid_url).is_err());

            // Test empty name
            let cmd_empty_name = RemoteCommand::Add {
                name: "".to_string(),
                url: "https://genhub.bio/user/repo.gen".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_empty_name).is_err());

            // Test empty URL
            let cmd_empty_url = RemoteCommand::Add {
                name: "origin".to_string(),
                url: "".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_empty_url).is_err());
        }

        #[test]
        fn test_remote_list_command() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Test list with no remotes
            let cmd_list = RemoteCommand::List;
            assert!(handle_remote_command(op_conn, &cmd_list).is_ok());

            // Add some remotes
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            Remote::create(op_conn, "upstream", "https://genhub.bio/upstream/repo.gen").unwrap();

            // Test list with remotes
            assert!(handle_remote_command(op_conn, &cmd_list).is_ok());
        }

        #[test]
        fn test_remote_remove_command() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Add a remote first
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Test successful remove
            let cmd_remove = RemoteCommand::Remove {
                name: "origin".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_remove).is_ok());

            // Verify remote was removed
            assert!(Remote::get_by_name_optional(op_conn, "origin").is_none());

            // Test remove non-existent remote
            let cmd_remove_missing = RemoteCommand::Remove {
                name: "nonexistent".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_remove_missing).is_err());
        }

        #[test]
        fn test_remote_remove_clears_default() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Add a remote and set it as default
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            Defaults::set_default_remote(op_conn, Some("origin")).unwrap();

            // Verify default is set
            assert_eq!(
                Defaults::get_default_remote(op_conn),
                Some("origin".to_string())
            );

            // Remove the remote
            let cmd_remove = RemoteCommand::Remove {
                name: "origin".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_remove).is_ok());

            // Verify default was cleared
            assert_eq!(Defaults::get_default_remote(op_conn), None);
        }

        #[test]
        fn test_remote_set_default_command() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Add a remote first
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // Test successful set default
            let cmd_set_default = RemoteCommand::SetDefault {
                name: "origin".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_set_default).is_ok());

            // Verify default was set
            assert_eq!(
                Defaults::get_default_remote(op_conn),
                Some("origin".to_string())
            );

            // Test set default for non-existent remote
            let cmd_set_default_missing = RemoteCommand::SetDefault {
                name: "nonexistent".to_string(),
            };
            assert!(handle_remote_command(op_conn, &cmd_set_default_missing).is_err());
        }

        #[test]
        fn test_remote_get_default_command() {
            let context = setup_gen();
            let op_conn = context.config().conn();

            // Test get default when none is set
            let cmd_get_default = RemoteCommand::GetDefault;
            assert!(handle_remote_command(op_conn, &cmd_get_default).is_ok());

            // Add a remote and set it as default
            Remote::create(op_conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            Defaults::set_default_remote(op_conn, Some("origin")).unwrap();

            // Test get default when one is set
            assert!(handle_remote_command(op_conn, &cmd_get_default).is_ok());
        }

        #[test]
        fn test_discovery_endpoint_from_genhub_repo_url() {
            assert_eq!(
                discovery_endpoint("https://genhub.bio/api/repos/david/example"),
                Some("https://genhub.bio/api/repos/david/example/remote".to_string())
            );
            assert_eq!(
                discovery_endpoint("https://genhub.bio/api/repos/david/example/"),
                Some("https://genhub.bio/api/repos/david/example/remote".to_string())
            );
        }

        #[test]
        fn test_discovery_endpoint_skips_existing_dolt_url() {
            assert_eq!(
                discovery_endpoint("https://genhub.bio/api/repos/david/example/dolt"),
                None
            );
        }

        #[test]
        fn test_discover_dolt_remote_url_uses_genhub_remote_endpoint() {
            let listener = TcpListener::bind("127.0.0.1:0").expect("should bind test listener");
            let address = listener
                .local_addr()
                .expect("should read test listener address");
            let server = thread::spawn(move || {
                let (mut stream, _) = listener.accept().expect("should accept request");
                let mut buffer = [0_u8; 2048];
                let bytes_read = stream.read(&mut buffer).expect("should read request");
                let request = String::from_utf8_lossy(&buffer[..bytes_read]);
                assert!(
                    request.starts_with("GET /api/repos/david/example/remote HTTP/1.1"),
                    "unexpected request: {request}"
                );

                let body = r#"{"remote":{"url":"http://127.0.0.1/dolt-endpoint"}}"#;
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .expect("should write response");
            });

            let remote_url = format!("http://{address}/api/repos/david/example");
            let discovered_url =
                discover_dolt_remote_url(&remote_url).expect("should resolve discovery URL");
            server.join().expect("should finish test server");

            assert_eq!(
                discovered_url,
                Some("http://127.0.0.1/dolt-endpoint".to_string())
            );
        }

        #[test]
        fn test_validate_dolt_remote_url_accepts_file_and_http() {
            assert!(validate_dolt_remote_url("file:///tmp/repo/.gen/default.db").is_ok());
            assert!(validate_dolt_remote_url("http://127.0.0.1:9000/repo").is_ok());
        }

        #[test]
        fn test_validate_dolt_remote_url_rejects_https() {
            let err = validate_dolt_remote_url("https://genhub.bio/api/repos/david/example/dolt")
                .expect_err("should reject https remotes");
            assert!(
                err.to_string().contains(
                    "Supported remote schemes are `file://` and unauthenticated `http://` only"
                ),
                "unexpected error: {err}"
            );
        }
    }
}
