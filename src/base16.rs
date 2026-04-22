use std::{path::PathBuf, str::FromStr};

use figment::{
    Figment,
    providers::{Format, Yaml},
};
use ratatui::style::Color;
use serde::{Deserialize, de, de::Deserializer};
use serde_with::serde_as;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Base16PaletteError {
    #[error("unable to extract data from file")]
    ExtractionFailed(#[from] figment::Error),
}

#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Base16Palette {
    #[serde(skip, alias = "scheme")]
    pub name: &'static str,
    #[serde(skip)]
    pub author: &'static str,
    #[serde(skip)]
    pub slug: &'static str,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base00: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base01: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base02: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base03: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base04: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base05: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base06: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base07: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base08: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base09: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0a: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0b: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0c: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0d: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0e: Color,
    #[serde(deserialize_with = "deserialize_from_str")]
    pub base0f: Color,
}

impl Default for Base16Palette {
    fn default() -> Self {
        Self {
            name: "Default",
            author: "Default",
            slug: "default",
            base00: Color::Indexed(0),
            base01: Color::Indexed(1),
            base02: Color::Indexed(2),
            base03: Color::Indexed(3),
            base04: Color::Indexed(4),
            base05: Color::Indexed(5),
            base06: Color::Indexed(6),
            base07: Color::Indexed(7),
            base08: Color::Indexed(8),
            base09: Color::Indexed(9),
            base0a: Color::Indexed(10),
            base0b: Color::Indexed(11),
            base0c: Color::Indexed(12),
            base0d: Color::Indexed(13),
            base0e: Color::Indexed(14),
            base0f: Color::Indexed(15),
        }
    }
}

impl Base16Palette {
    #[allow(clippy::result_large_err)]
    pub fn from_yaml(file: impl Into<PathBuf>) -> Result<Self, Base16PaletteError> {
        Figment::new()
            .merge(Yaml::file(file.into()))
            .extract::<Base16Palette>()
            .map_err(Base16PaletteError::ExtractionFailed)
    }

    #[allow(clippy::result_large_err)]
    pub fn from_yaml_str(contents: &str) -> Result<Self, Base16PaletteError> {
        Figment::new()
            .merge(Yaml::string(contents))
            .extract::<Base16Palette>()
            .map_err(Base16PaletteError::ExtractionFailed)
    }
}

fn deserialize_from_str<'de, D>(deserializer: D) -> Result<Color, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    if s.starts_with('#') {
        Color::from_str(&s).map_err(de::Error::custom)
    } else {
        Color::from_str(&format!("#{s}")).map_err(de::Error::custom)
    }
}
