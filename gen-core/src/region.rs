use std::num::ParseIntError;

use thiserror::Error;

#[derive(Debug, PartialEq)]
pub struct Region {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Error, PartialEq)]
pub enum RegionParseError {
    #[error("Region start is less than region end")]
    InvalidRange,
    #[error("Parsing error: Region name not present")]
    NoName,
    #[error("Parsing error: Region coordinates not present")]
    NoCoordinates,
    #[error("Parsing error: Start coordinate not present")]
    NoStartCoordinate,
    #[error("Parsing error: End coordinate not present")]
    NoEndCoordinate,
    #[error("Parsing error: Invalid coordinate: {0}")]
    InvalidCoordinate(#[from] ParseIntError),
}

impl Region {
    pub fn parse(region_string: &str, zero_based: bool) -> Result<Self, RegionParseError> {
        // Example input: "chr1:100-200"
        let mut parts = region_string.split(':');
        let name = parts.next().ok_or(RegionParseError::NoName)?.to_string();
        let interval = parts.next().ok_or(RegionParseError::NoCoordinates)?;
        let mut bounds = interval.split('-');
        let start = bounds
            .next()
            .ok_or(RegionParseError::NoStartCoordinate)?
            .parse::<i64>()
            .map_err(RegionParseError::InvalidCoordinate)?;
        let end = bounds
            .next()
            .ok_or(RegionParseError::NoEndCoordinate)?
            .parse::<i64>()
            .map_err(RegionParseError::InvalidCoordinate)?;
        if start > end {
            return Err(RegionParseError::InvalidRange);
        }

        let increment = if zero_based { 0 } else { 1 };

        Ok(Self {
            name,
            start: start + increment,
            end: end + increment,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_region() {
        let region = Region::parse("chr1:100-200", true);
        assert_eq!(
            region,
            Ok(Region {
                name: "chr1".to_string(),
                start: 100,
                end: 200,
            })
        );
    }

    #[test]
    fn test_parse_invalid_format() {
        let region = Region::parse("chr1-100-200", true);
        assert_eq!(region, Err(RegionParseError::NoCoordinates));
    }

    #[test]
    fn test_parse_start_greater_than_end() {
        let region = Region::parse("chr1:300-200", true);
        assert_eq!(region, Err(RegionParseError::InvalidRange));
    }

    #[test]
    fn test_region_applies_to_array_slice() {
        let string = "foobarbaz";
        let region = Region::parse("chr1:3-5", true).unwrap();
        let slice = &string[region.start as usize..region.end as usize];
        assert_eq!(slice, "ba");
    }
}
