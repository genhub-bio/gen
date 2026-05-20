use std::num::ParseIntError;

use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Region {
    pub name: String,
    pub start: Option<i64>,
    pub end: Option<i64>,
}

#[derive(Debug, Error, PartialEq)]
pub enum RegionParseError {
    #[error("Region start is less than region end")]
    InvalidRange,
    #[error("Parsing error: Region name not present")]
    NoName,
    #[error("Parsing error: Invalid coordinate syntax")]
    InvalidSyntax,
    #[error("Parsing error: Region coordinates not present")]
    NoCoordinates,
    #[error("Parsing error: Start coordinate not present")]
    NoStartCoordinate,
    #[error("Parsing error: End coordinate not present")]
    NoEndCoordinate,
    #[error("Parsing error: Invalid coordinate: {0}")]
    InvalidCoordinate(#[from] ParseIntError),
}

#[derive(Debug, Error)]
pub enum RegionResolutionError<E: std::error::Error + 'static> {
    #[error("Region not found: {0}")]
    NotFound(String),
    #[error("Region is ambiguous: {0}")]
    Ambiguous(String),
    #[error(transparent)]
    Lookup(#[from] E),
}

pub trait RegionResolver: Sized {
    type Connection;
    type Error: std::error::Error + 'static;

    fn resolve(
        region: &Region,
        conn: &Self::Connection,
        collection_name: &str,
        sample_name: &str,
    ) -> Result<Self, RegionResolutionError<Self::Error>>;
}

impl Region {
    /// Parse a region string.
    ///
    /// Supported forms:
    /// - `name`
    /// - `name:position`
    /// - `name:start-end`
    /// - `name:start..`
    /// - `name:..end`
    /// - `name:..`
    ///
    /// Examples:
    /// - `chr1`
    /// - `chr1:100`
    /// - `chr1:100-200`
    /// - `chr1:100..`
    /// - `chr1:..200`
    /// - `mreB:-35--10`
    ///
    /// A single position like `chr1:100` is parsed as an exact point with
    /// `start == end == 100`. Open-ended ranges must use `..`.
    pub fn parse(region_string: &str) -> Result<Self, RegionParseError> {
        let (name, coordinates) = match region_string.split_once(':') {
            Some((name, coordinates)) => (name.trim(), Some(coordinates.trim())),
            None => (region_string.trim(), None),
        };

        if name.is_empty() {
            return Err(RegionParseError::NoName);
        }

        let (start, end) = match coordinates {
            Some(coordinates) => {
                if coordinates.is_empty() {
                    return Err(RegionParseError::InvalidSyntax);
                }

                if let Some((start, end)) = coordinates.split_once("..") {
                    let start = if start.is_empty() {
                        None
                    } else {
                        Some(
                            start
                                .parse::<i64>()
                                .map_err(RegionParseError::InvalidCoordinate)?,
                        )
                    };
                    let end = if end.is_empty() {
                        None
                    } else {
                        Some(
                            end.parse::<i64>()
                                .map_err(RegionParseError::InvalidCoordinate)?,
                        )
                    };
                    if start.is_none() && end.is_none() {
                        (None, None)
                    } else {
                        (start, end)
                    }
                } else {
                    let bytes = coordinates.as_bytes();
                    let separator = bytes
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find_map(|(index, byte)| (*byte == b'-').then_some(index));

                    match separator {
                        Some(index) => {
                            let start = coordinates[..index]
                                .parse::<i64>()
                                .map_err(RegionParseError::InvalidCoordinate)?;
                            let end = coordinates[(index + 1)..]
                                .parse::<i64>()
                                .map_err(RegionParseError::InvalidCoordinate)?;
                            (Some(start), Some(end))
                        }
                        None => {
                            let position = coordinates
                                .parse::<i64>()
                                .map_err(RegionParseError::InvalidCoordinate)?;
                            (Some(position), Some(position))
                        }
                    }
                }
            }
            None => (None, None),
        };

        Ok(Self {
            name: name.to_string(),
            start,
            end,
        })
    }

    pub fn require_coordinates(&self) -> Result<(i64, i64), RegionParseError> {
        let start = self.start.ok_or(RegionParseError::NoCoordinates)?;
        let end = self.end.ok_or(RegionParseError::NoEndCoordinate)?;
        Ok((start, end))
    }
}

impl std::fmt::Display for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.start, self.end) {
            (None, None) => write!(f, "{}", self.name),
            (Some(start), Some(end)) if start == end => write!(f, "{}:{start}", self.name),
            (Some(start), None) => write!(f, "{}:{start}..", self.name),
            (None, Some(end)) => write!(f, "{}:..{end}", self.name),
            (Some(start), Some(end)) => write!(f, "{}:{start}-{end}", self.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_region() {
        let region = Region::parse("chr1:100-200");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: Some(100),
                end: Some(200),
            })
        );
    }

    #[test]
    fn test_parse_name_only() {
        let region = Region::parse("chr1");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: None,
                end: None,
            })
        );
    }

    #[test]
    fn test_parse_start_only() {
        let region = Region::parse("chr1:100");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: Some(100),
                end: Some(100),
            })
        );
    }

    #[test]
    fn test_parse_to_end_syntax() {
        let region = Region::parse("chr1:100..");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: Some(100),
                end: None,
            })
        );
    }

    #[test]
    fn test_parse_to_start_syntax() {
        let region = Region::parse("chr1:..100");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: None,
                end: Some(100),
            })
        );
    }

    #[test]
    fn test_parse_entire_region_with_open_syntax() {
        let region = Region::parse("chr1:..");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: None,
                end: None,
            })
        );
    }

    #[test]
    fn test_parse_negative_range() {
        let region = Region::parse("mreB:-35--10");
        assert_eq!(
            region,
            Ok(Region {
                name: "mreB".to_string(),
                start: Some(-35),
                end: Some(-10),
            })
        );
    }

    #[test]
    fn test_parse_invalid_syntax() {
        let region = Region::parse("chr1:");
        assert_eq!(region, Err(RegionParseError::InvalidSyntax));
    }

    #[test]
    fn test_parse_start_greater_than_end() {
        let region = Region::parse("chr1:300-200");
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: Some(300),
                end: Some(200),
            })
        );
    }

    #[test]
    fn test_region_applies_to_array_slice() {
        let string = "foobarbaz";
        let region = Region::parse("chr1:3-5").unwrap();
        let (start, end) = region.require_coordinates().unwrap();
        let slice = &string[start as usize..end as usize];
        assert_eq!(slice, "ba");
    }
}
