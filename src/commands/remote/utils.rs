use std::{
    fs::{File, create_dir_all},
    io::Write,
    path::{Path, PathBuf},
};

use base64::{
    Engine as _,
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
};
use directories::ProjectDirs;
use getrandom;

use crate::commands::remote::server::AuthTokens;

pub fn generate_state() -> Result<String, getrandom::Error> {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf)?;

    Ok(URL_SAFE.encode(buf))
}

fn get_token_path(identity: &str) -> PathBuf {
    if let Some(proj_dirs) = ProjectDirs::from("org", "gen", "gen") {
        let key = URL_SAFE_NO_PAD.encode(identity.as_bytes());
        let dir = proj_dirs.config_dir().join("remotes").join(key);
        create_dir_all(&dir).expect("Unable to create config dir");
        dir.join("tokens.json")
    } else {
        PathBuf::from(format!(
            "{}_tokens.json",
            URL_SAFE_NO_PAD.encode(identity.as_bytes())
        ))
    }
}

pub fn save_tokens(identity: &str, tokens: &AuthTokens) -> std::io::Result<()> {
    let path = get_token_path(identity);
    save_tokens_to_path(&path, tokens)
}

fn save_tokens_to_path(path: &Path, tokens: &AuthTokens) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(tokens)?;
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn load_tokens(identity: &str) -> std::io::Result<AuthTokens> {
    let path = get_token_path(identity);
    let file = File::open(&path)?;
    let tokens: AuthTokens = serde_json::from_reader(file)?;
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod generate_state {
        use base64::engine::general_purpose::URL_SAFE;

        use super::*;

        #[test]
        fn test_generate_state_ok_and_format() {
            let state = generate_state().expect("Failed to generate state");

            // Base64 URL_SAFE encoding of 32 bytes should be 44 characters w/ padding
            assert_eq!(state.len(), 44, "State should be 44 chars long");

            // Decode back to ensure it's valid base64 URL_SAFE
            let decoded = URL_SAFE
                .decode(&state)
                .expect("State should be valid base64 URL_SAFE");

            assert_eq!(decoded.len(), 32, "Decoded state should be 32 bytes long");
        }

        #[test]
        fn test_generate_state_multiple_unique() {
            let s1 = generate_state().unwrap();
            let s2 = generate_state().unwrap();
            assert_ne!(s1, s2, "Two consecutive states should not be equal");
        }
    }
}
