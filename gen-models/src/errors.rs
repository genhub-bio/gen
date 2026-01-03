use gen_core::errors::{ConfigError, ConnectionError, StrandError};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, Hash, PartialEq)]
pub enum QueryError {
    #[error("Results not found: {0}")]
    ResultsNotFound(String),
}

#[derive(Debug, Error, PartialEq)]
pub enum ChangeError {
    #[error("Operation Error: {0}")]
    OutOfBounds(String),
}

#[derive(Debug, Error, PartialEq)]
pub enum ChangesetError {
    #[error("Strand Error: {0}")]
    StrandError(#[from] StrandError),
    #[error("Missing Model: {0}")]
    MissingModel(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}

#[derive(Debug, PartialEq, Error)]
pub enum OperationError {
    #[error("Changeset Error: {0}")]
    ChangesetError(#[from] ChangesetError),
    #[error("Changeset Error: {0}")]
    ConnectionError(#[from] ConnectionError),
    #[error("No Changes")]
    NoChanges,
    #[error("Operation Already Exists")]
    OperationExists,
    #[error("Operation {0} Does Not Exist")]
    NoOperation(String),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
    #[error("Config Error: {0}")]
    ConfigError(#[from] ConfigError),
    #[error("Error storing data file")]
    IOError,
}

#[derive(Debug, PartialEq, Error)]
pub enum BranchError {
    #[error("Cannot delete branch: {0}")]
    CannotDelete(String),
    #[error("SQL Error: {0}")]
    SQLError(String),
    #[error("SQLite Error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}

#[derive(Debug, Error, PartialEq)]
pub enum RemoteError {
    #[error("Remote '{0}' already exists")]
    RemoteAlreadyExists(String),

    #[error("Remote '{0}' not found")]
    RemoteNotFound(String),

    #[error("Invalid remote URL: {0}")]
    InvalidUrl(String),

    #[error("Invalid remote name: {0}")]
    InvalidName(String),

    #[error("Remote name cannot be empty")]
    EmptyName,

    #[error("Remote URL cannot be empty")]
    EmptyUrl,

    #[error("Remote name can only contain letters, numbers, hyphens, and underscores")]
    InvalidNameCharacters,

    #[error("URL must use HTTP, HTTPS, SSH protocol, or be a valid file path")]
    UnsupportedUrlScheme,

    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    #[error("Cannot delete remote '{0}' as it is set as the default remote")]
    CannotDeleteDefaultRemote(String),

    #[error("Cannot set non-existent remote '{0}' as default")]
    DefaultRemoteNotFound(String),
}

#[derive(Debug, Error)]
pub enum FileAdditionError {
    #[error("Failed to read file: {0}")]
    FileReadError(#[from] std::io::Error),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Permission denied accessing file: {0}")]
    FilePermissionDenied(String),

    #[error("Failed to calculate checksum: {0}")]
    ChecksumError(String),

    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    #[error("HashId generation failed: {0}")]
    HashIdError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    mod remote_error_tests {
        use super::*;
        use crate::{
            db::OperationsConnection,
            operations::{Branch, Defaults, Remote},
            test_helpers::get_operation_connection,
        };

        fn setup_test_db() -> OperationsConnection {
            get_operation_connection(None).unwrap()
        }

        #[test]
        fn test_remote_validation_empty_name() {
            let result = Remote::validate_name("");
            assert_eq!(result, Err(RemoteError::EmptyName));
        }

        #[test]
        fn test_remote_validation_invalid_name_characters() {
            // Test various invalid characters
            assert_eq!(
                Remote::validate_name("remote with spaces"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote@special"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote.dot"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote/slash"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote\\backslash"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote:colon"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote;semicolon"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote|pipe"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote<bracket"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote>bracket"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote?question"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote*asterisk"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote+plus"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote=equals"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote[bracket"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote]bracket"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote{brace"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote}brace"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote(paren"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote)paren"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote\"quote"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote'quote"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote`backtick"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote~tilde"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote!exclamation"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote#hash"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote$dollar"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote%percent"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote^caret"),
                Err(RemoteError::InvalidNameCharacters)
            );
            assert_eq!(
                Remote::validate_name("remote&ampersand"),
                Err(RemoteError::InvalidNameCharacters)
            );
        }

        #[test]
        fn test_remote_validation_valid_names() {
            // Test valid names
            assert!(Remote::validate_name("origin").is_ok());
            assert!(Remote::validate_name("my-remote").is_ok());
            assert!(Remote::validate_name("remote_1").is_ok());
            assert!(Remote::validate_name("test123").is_ok());
            assert!(Remote::validate_name("UPPERCASE").is_ok());
            assert!(Remote::validate_name("MixedCase").is_ok());
            assert!(Remote::validate_name("remote-with-many-hyphens").is_ok());
            assert!(Remote::validate_name("remote_with_many_underscores").is_ok());
            assert!(Remote::validate_name("remote123with456numbers").is_ok());
            assert!(Remote::validate_name("a").is_ok()); // Single character
            assert!(Remote::validate_name("a1").is_ok());
            assert!(Remote::validate_name("1a").is_ok());
            assert!(Remote::validate_name("123").is_ok()); // All numbers
            assert!(Remote::validate_name("remote-123_test").is_ok()); // Mixed valid characters
        }

        #[test]
        fn test_url_validation_empty_url() {
            let result = Remote::validate_url("");
            assert_eq!(result, Err(RemoteError::EmptyUrl));
        }

        #[test]
        fn test_url_validation_valid_urls() {
            // Valid HTTP/HTTPS URLs
            assert!(Remote::validate_url("https://genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("http://example.com/repo").is_ok());
            assert!(Remote::validate_url("https://gitlab.com/user/repo").is_ok());
            assert!(Remote::validate_url("http://localhost:8080/repo").is_ok());

            // Valid SSH URLs
            assert!(Remote::validate_url("ssh://git@genhub.bio/user/repo.gen").is_ok());
            assert!(Remote::validate_url("ssh://user@host.com:2222/path/to/repo").is_ok());

            // Valid directories
            assert!(Remote::validate_url("/path/to/local/repo").is_ok());
            assert!(Remote::validate_url("/home/user/repos/myrepo").is_ok());
            assert!(Remote::validate_url("/tmp/test").is_ok());
            assert!(Remote::validate_url("file:///tmp/test").is_ok());

            // Valid SSH-style paths
            assert!(Remote::validate_url("user@host:path/to/repo").is_ok());
            assert!(Remote::validate_url("git@genhub.bio:user/repo.gen").is_ok());
            assert!(Remote::validate_url("user@server.com:~/repos/project").is_ok());
        }

        #[test]
        fn test_url_validation_invalid_urls() {
            // Invalid scheme
            assert_eq!(
                Remote::validate_url("ftp://invalid-protocol.com"),
                Err(RemoteError::UnsupportedUrlScheme)
            );
            assert_eq!(
                Remote::validate_url("smb://network/share"),
                Err(RemoteError::UnsupportedUrlScheme)
            );

            // Invalid format
            assert_eq!(
                Remote::validate_url("not-a-url"),
                Err(RemoteError::UnsupportedUrlScheme)
            );
            assert_eq!(
                Remote::validate_url("just-text"),
                Err(RemoteError::UnsupportedUrlScheme)
            );
            assert_eq!(
                Remote::validate_url("no-scheme-or-path"),
                Err(RemoteError::UnsupportedUrlScheme)
            );

            // Malformed URLs with schemes
            assert!(matches!(
                Remote::validate_url("https://"),
                Err(RemoteError::InvalidUrl(_))
            ));
            assert!(matches!(
                Remote::validate_url("http://[invalid"),
                Err(RemoteError::InvalidUrl(_))
            ));
        }

        #[test]
        fn test_remote_create_success() {
            let conn = setup_test_db();

            let result = Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen");
            assert!(result.is_ok());

            let remote = result.unwrap();
            assert_eq!(remote.name, "origin");
            assert_eq!(remote.url, "https://genhub.bio/user/repo.gen");
        }

        #[test]
        fn test_remote_create_duplicate_name() {
            let conn = setup_test_db();

            // Create first remote
            Remote::create(&conn, "origin", "https://genhub.bio/user/repo1.gen").unwrap();

            // Try to create duplicate
            let result = Remote::create(&conn, "origin", "https://genhub.bio/user/repo2.gen");
            assert_eq!(
                result,
                Err(RemoteError::RemoteAlreadyExists("origin".to_string()))
            );
        }

        #[test]
        fn test_remote_create_invalid_name() {
            let conn = setup_test_db();

            let result = Remote::create(&conn, "", "https://genhub.bio/user/repo.gen");
            assert_eq!(result, Err(RemoteError::EmptyName));

            let result = Remote::create(&conn, "invalid name", "https://genhub.bio/user/repo.gen");
            assert_eq!(result, Err(RemoteError::InvalidNameCharacters));
        }

        #[test]
        fn test_remote_create_invalid_url() {
            let conn = setup_test_db();

            let result = Remote::create(&conn, "origin", "");
            assert_eq!(result, Err(RemoteError::EmptyUrl));

            let result = Remote::create(&conn, "origin", "ftp://invalid.com");
            assert_eq!(result, Err(RemoteError::UnsupportedUrlScheme));
        }

        #[test]
        fn test_remote_get_by_name_success() {
            let conn = setup_test_db();

            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            let result = Remote::get_by_name(&conn, "origin");
            assert!(result.is_ok());

            let remote = result.unwrap();
            assert_eq!(remote.name, "origin");
            assert_eq!(remote.url, "https://genhub.bio/user/repo.gen");
        }

        #[test]
        fn test_remote_get_by_name_not_found() {
            let conn = setup_test_db();

            let result = Remote::get_by_name(&conn, "nonexistent");
            assert_eq!(
                result,
                Err(RemoteError::RemoteNotFound("nonexistent".to_string()))
            );
        }

        #[test]
        fn test_remote_delete_success() {
            let conn = setup_test_db();

            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            let result = Remote::delete(&conn, "origin");
            assert!(result.is_ok());

            // Verify it's deleted
            let get_result = Remote::get_by_name(&conn, "origin");
            assert_eq!(
                get_result,
                Err(RemoteError::RemoteNotFound("origin".to_string()))
            );
        }

        #[test]
        fn test_remote_delete_not_found() {
            let conn = setup_test_db();

            let result = Remote::delete(&conn, "nonexistent");
            assert_eq!(
                result,
                Err(RemoteError::RemoteNotFound("nonexistent".to_string()))
            );
        }

        #[test]
        fn test_remote_exists() {
            let conn = setup_test_db();

            assert!(!Remote::exists(&conn, "origin"));

            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            assert!(Remote::exists(&conn, "origin"));

            Remote::delete(&conn, "origin").unwrap();
            assert!(!Remote::exists(&conn, "origin"));
        }

        #[test]
        fn test_defaults_set_default_remote_success() {
            let conn = setup_test_db();

            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            let result = Defaults::set_default_remote(&conn, Some("origin"));
            assert!(result.is_ok());

            let default = Defaults::get_default_remote(&conn);
            assert_eq!(default, Some("origin".to_string()));
        }

        #[test]
        fn test_defaults_set_default_remote_not_found() {
            let conn = setup_test_db();

            let result = Defaults::set_default_remote(&conn, Some("nonexistent"));
            assert_eq!(
                result,
                Err(RemoteError::RemoteNotFound("nonexistent".to_string()))
            );
        }

        #[test]
        fn test_defaults_set_default_remote_clear() {
            let conn = setup_test_db();

            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            Defaults::set_default_remote(&conn, Some("origin")).unwrap();

            let result = Defaults::set_default_remote(&conn, None);
            assert!(result.is_ok());

            let default = Defaults::get_default_remote(&conn);
            assert_eq!(default, None);
        }

        #[test]
        fn test_defaults_get_default_remote_url() {
            let conn = setup_test_db();

            // No default set
            assert_eq!(Defaults::get_default_remote_url(&conn), None);

            // Set default
            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();
            Defaults::set_default_remote(&conn, Some("origin")).unwrap();

            let url = Defaults::get_default_remote_url(&conn);
            assert_eq!(url, Some("https://genhub.bio/user/repo.gen".to_string()));

            // Delete remote (should return None even though default is set)
            Remote::delete(&conn, "origin").unwrap();
            let url = Defaults::get_default_remote_url(&conn);
            assert_eq!(url, None);
        }

        #[test]
        fn test_branch_set_remote_validated_success() {
            let conn = setup_test_db();

            // Create a remote and branch
            Remote::create(&conn, "origin", "https://genhub.bio/user/repo.gen").unwrap();

            // For this test, we'll assume branch ID 1 exists (this would be set up by the test framework)
            // In a real scenario, you'd create a branch first
            let branch_id = 1i64;

            let result = Branch::set_remote_validated(&conn, branch_id, Some("origin"));
            assert!(result.is_ok());
        }

        #[test]
        fn test_branch_set_remote_validated_remote_not_found() {
            let conn = setup_test_db();

            let branch_id = 1i64;

            let result = Branch::set_remote_validated(&conn, branch_id, Some("nonexistent"));
            assert_eq!(
                result,
                Err(RemoteError::RemoteNotFound("nonexistent".to_string()))
            );
        }

        #[test]
        fn test_branch_set_remote_validated_clear() {
            let conn = setup_test_db();

            let branch_id = 1i64;

            let result = Branch::set_remote_validated(&conn, branch_id, None);
            assert!(result.is_ok());
        }

        #[test]
        fn test_error_display_messages() {
            // Test that error messages are user-friendly
            assert_eq!(
                RemoteError::EmptyName.to_string(),
                "Remote name cannot be empty"
            );
            assert_eq!(
                RemoteError::EmptyUrl.to_string(),
                "Remote URL cannot be empty"
            );
            assert_eq!(
                RemoteError::InvalidNameCharacters.to_string(),
                "Remote name can only contain letters, numbers, hyphens, and underscores"
            );
            assert_eq!(
                RemoteError::UnsupportedUrlScheme.to_string(),
                "URL must use HTTP, HTTPS, SSH protocol, or be a valid file path"
            );
            assert_eq!(
                RemoteError::RemoteAlreadyExists("origin".to_string()).to_string(),
                "Remote 'origin' already exists"
            );
            assert_eq!(
                RemoteError::RemoteNotFound("origin".to_string()).to_string(),
                "Remote 'origin' not found"
            );
            assert_eq!(
                RemoteError::InvalidUrl("test".to_string()).to_string(),
                "Invalid remote URL: test"
            );
            assert_eq!(
                RemoteError::InvalidName("test".to_string()).to_string(),
                "Invalid remote name: test"
            );
            assert_eq!(
                RemoteError::CannotDeleteDefaultRemote("origin".to_string()).to_string(),
                "Cannot delete remote 'origin' as it is set as the default remote"
            );
            assert_eq!(
                RemoteError::DefaultRemoteNotFound("origin".to_string()).to_string(),
                "Cannot set non-existent remote 'origin' as default"
            );
        }

        #[test]
        fn test_error_equality() {
            // Test that errors can be compared for equality
            assert_eq!(RemoteError::EmptyName, RemoteError::EmptyName);
            assert_eq!(RemoteError::EmptyUrl, RemoteError::EmptyUrl);
            assert_eq!(
                RemoteError::InvalidNameCharacters,
                RemoteError::InvalidNameCharacters
            );
            assert_eq!(
                RemoteError::UnsupportedUrlScheme,
                RemoteError::UnsupportedUrlScheme
            );
            assert_eq!(
                RemoteError::RemoteAlreadyExists("test".to_string()),
                RemoteError::RemoteAlreadyExists("test".to_string())
            );
            assert_eq!(
                RemoteError::RemoteNotFound("test".to_string()),
                RemoteError::RemoteNotFound("test".to_string())
            );

            // Test inequality
            assert_ne!(RemoteError::EmptyName, RemoteError::EmptyUrl);
            assert_ne!(
                RemoteError::RemoteAlreadyExists("test1".to_string()),
                RemoteError::RemoteAlreadyExists("test2".to_string())
            );
        }

        #[test]
        fn test_comprehensive_validation_edge_cases() {
            // Test edge cases for name validation
            assert!(Remote::validate_name("a").is_ok()); // Single character
            assert!(Remote::validate_name("1").is_ok()); // Single digit
            assert!(Remote::validate_name("-").is_ok()); // Single hyphen
            assert!(Remote::validate_name("_").is_ok()); // Single underscore

            // Test very long names (should be valid as long as they contain valid characters)
            let long_name = "a".repeat(1000);
            assert!(Remote::validate_name(&long_name).is_ok());

            let long_name_with_valid_chars = "a-b_c".repeat(200);
            assert!(Remote::validate_name(&long_name_with_valid_chars).is_ok());

            // Test edge cases for URL validation
            assert!(Remote::validate_url("https://a.b").is_ok()); // Minimal valid URL
            assert!(Remote::validate_url("/a").is_ok()); // Minimal file path
            assert!(Remote::validate_url("a:b").is_ok()); // Minimal SSH-style path
        }
    }
}
