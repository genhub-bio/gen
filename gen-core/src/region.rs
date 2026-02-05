#[derive(Debug, PartialEq)]
pub struct Region {
    pub name: String,
    pub start: i64,
    pub end: i64,
}

impl Region {
    pub fn parse(s: &str) -> Option<Self> {
        // Example input: "chr1:100-200"
        let mut parts = s.split(':');
        let name = parts.next()?.to_string();
        let interval = parts.next()?;
        let mut bounds = interval.split('-');
        let start = bounds.next()?.parse::<i64>().ok()?;
        let end = bounds.next()?.parse::<i64>().ok()?;
        if start > end {
            return None;
        }
        Some(Self { name, start, end })
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
            Some(Region {
                name: "chr1".to_string(),
                start: 100,
                end: 200,
            })
        );
    }

    #[test]
    fn test_parse_invalid_format() {
        let region = Region::parse("chr1-100-200");
        assert_eq!(region, None);
    }

    #[test]
    fn test_parse_start_greater_than_end() {
        let region = Region::parse("chr1:300-200");
        assert_eq!(region, None);
    }

    #[test]
    fn test_region_applies_to_array_slice() {
        let string = "foobarbaz";
        let region = Region::parse("chr1:3-5").unwrap();
        let slice = &string[region.start as usize..region.end as usize];
        assert_eq!(slice, "ba");
    }
}
