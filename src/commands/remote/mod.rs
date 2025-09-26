use clap::Subcommand;
use gen_models::operations::{Defaults, Remote};
use reqwest::{blocking::Client, redirect::Policy};
use rusqlite::Connection;

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

/// Handle remote management commands with comprehensive error handling
pub fn handle_remote_command(
    conn: &Connection,
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
                Err(remote_err.into())
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

        RemoteCommand::Remove { name } => {
            // Check if this is the default remote and clear it if so
            if let Some(default_remote) = Defaults::get_default_remote(conn)
                && default_remote == *name
                && let Err(err) = Defaults::set_default_remote_compat(conn, None)
            {
                eprintln!("Warning: Failed to clear default remote: {err}");
            }

            // Delete the remote (this will set branch remote associations to null via foreign key constraint)
            match Remote::delete(conn, name) {
                Ok(_) => {
                    println!("Remote '{name}' removed successfully");
                    Ok(())
                }
                Err(remote_err) => {
                    eprintln!("Error: {remote_err}");
                    Err(remote_err.into())
                }
            }
        }

        RemoteCommand::SetDefault { name } => {
            match Defaults::set_default_remote(conn, Some(name)) {
                Ok(_) => {
                    println!("Default remote set to '{name}'");
                    Ok(())
                }
                Err(remote_err) => {
                    eprintln!("Error: {remote_err}");
                    Err(remote_err.into())
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

        RemoteCommand::Login { name } => {
            let remote_name = name
                .clone()
                .ok_or_else(|| Defaults::get_default_remote(conn))
                .expect("No remote specified and no default set.");
            let remote = Remote::get_by_name(conn, &remote_name).expect("Unable to find remote.");
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
            let (local_addr, handle, rx) = server::start_callback_server(state.clone())
                .expect("Unable to start callback server.");

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

                // Open in default browser
                webbrowser::open(redirect_url)?;
            } else {
                println!("No redirect URL found. Response: {res:?}");
            }
            // Wait for the callback handler to complete
            handle.join().unwrap();

            if let Ok(tokens) = rx.recv() {
                utils::save_tokens(&remote_name, &tokens)
                    .expect("Failed to save login information.");
            }

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{get_operation_connection, setup_gen_dir};

    #[cfg(test)]
    mod remote {
        use super::*;

        #[test]
        fn test_remote_add_command() {
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
            setup_gen_dir();
            let op_conn = &get_operation_connection(None).unwrap();

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
