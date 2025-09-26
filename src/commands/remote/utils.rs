use std::{
    fs::{File, create_dir_all},
    io::Write,
    path::PathBuf,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE};
use directories::ProjectDirs;
use getrandom;

use crate::commands::remote::server::AuthTokens;

pub fn generate_state() -> Result<String, getrandom::Error> {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf)?;

    Ok(URL_SAFE.encode(buf))
}

fn get_token_path(remote: &str) -> PathBuf {
    if let Some(proj_dirs) = ProjectDirs::from("org", "gen", "gen") {
        let dir = proj_dirs.config_dir().join(remote);
        create_dir_all(&dir).expect("Unable to create config dir");
        dir.join("tokens.json")
    } else {
        PathBuf::from(format!("{remote}_tokens.json"))
    }
}

pub fn save_tokens(remote: &str, tokens: &AuthTokens) -> std::io::Result<()> {
    let path = get_token_path(remote);

    let json = serde_json::to_string_pretty(tokens)?;

    let mut file = File::create(&path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn load_tokens(remote: &str) -> std::io::Result<AuthTokens> {
    let path = get_token_path(remote);
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
