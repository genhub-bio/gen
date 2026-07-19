use clap::Subcommand;
use gen_models::{
    db::ConfigConnection,
    operations::{Defaults, Remote},
};
use thiserror::Error;

use crate::commands::remote::server::AuthTokens;

// Unconditional (not `#[cfg(target_os = "emscripten")]`) so its pure URL/message-framing logic
// gets exercised by the normal native test run too; only its actual stdin/stdout I/O is gated.
pub mod browser;
#[cfg(not(target_os = "emscripten"))]
pub mod client;
#[cfg(not(target_os = "emscripten"))]
pub mod operations;
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

/// Logs in via the native flow: starts a localhost callback listener, opens the GenHub login URL
/// in the system browser, and blocks until the listener receives the callback. Unavailable on
/// Emscripten (no TCP sockets, no system browser to open); see the Emscripten variant below.
#[cfg(not(target_os = "emscripten"))]
pub fn login_origin(origin: &str) -> Result<AuthTokens, Box<dyn std::error::Error>> {
    use reqwest::{blocking::Client, redirect::Policy};

    println!("Logging in to remote: {origin}");
    let state = utils::generate_state().expect("Unable to generate random nonce.");
    let (local_addr, handle, rx) =
        server::start_callback_server(state.clone()).expect("Unable to start callback server.");

    let client = Client::builder().redirect(Policy::none()).build()?;
    let res = client
        .get(format!("{origin}/api/auth/cli/login/"))
        .query(&[
            ("redirect_uri", &format!("{origin}/api/auth/cli/callback")),
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
        return Err(Box::new(RemoteError::NoRedirectUrl(origin.to_string())));
    }

    handle.join().map_err(|_| "Login callback thread failed")?;
    rx.recv()
        .map_err(|error| Box::new(error) as Box<dyn std::error::Error>)
}

/// Logs in via the browser flow: the terminal opens the GenHub login URL in a popup (falling
/// back to a clickable link if the popup is blocked), and waits for the browser callback page to
/// deliver the result through the Cockle host bridge. See `browser::login_origin` for details.
#[cfg(target_os = "emscripten")]
pub fn login_origin(origin: &str) -> Result<AuthTokens, Box<dyn std::error::Error>> {
    browser::login_origin(origin)
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
    let origin = utils::normalized_origin(&remote.url)?;
    let tokens = login_origin(&origin)?;
    utils::save_tokens(&origin, &tokens).expect("Failed to save login information.");

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
    }
}
