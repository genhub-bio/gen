use noodles::gff::record::Strand as GFFStrand;
use rusqlite::types::{FromSql, FromSqlResult, ToSqlOutput, Value, ValueRef};
use rusqlite::ToSql;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum Strand {
    Forward,
    Reverse,
    Unknown,
    ImportantButUnknown,
}

impl Strand {
    pub fn is_ambiguous(a: Strand) -> bool {
        a == Strand::ImportantButUnknown || a == Strand::Unknown
    }
    pub fn is_compatible(a: Strand, b: Strand) -> bool {
        let a_ambig = Strand::is_ambiguous(a);
        let b_ambig = Strand::is_ambiguous(b);
        (a_ambig || b_ambig) || a == b
    }
}

// example https://docs.rs/rusqlite/latest/rusqlite/types/index.html
impl ToSql for Strand {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let result = match self {
            Strand::Forward => "+".into(),
            Strand::Reverse => "-".into(),
            Strand::Unknown => ".".into(),
            Strand::ImportantButUnknown => "?".into(),
        };
        Ok(result)
    }
}

impl From<Strand> for Value {
    fn from(value: Strand) -> Value {
        let result = match value {
            Strand::Forward => "+",
            Strand::Reverse => "-",
            Strand::Unknown => ".",
            Strand::ImportantButUnknown => "?",
        };
        Value::Text(result.to_string())
    }
}

impl From<GFFStrand> for Strand {
    fn from(value: GFFStrand) -> Strand {
        match value {
            GFFStrand::Forward => Strand::Forward,
            GFFStrand::Unknown => Strand::Unknown,
            GFFStrand::Reverse => Strand::Reverse,
            GFFStrand::None => Strand::Unknown,
        }
    }
}

impl From<Strand> for GFFStrand {
    fn from(value: Strand) -> GFFStrand {
        match value {
            Strand::Forward => GFFStrand::Forward,
            Strand::Unknown => GFFStrand::Unknown,
            Strand::Reverse => GFFStrand::Reverse,
            Strand::Unknown => GFFStrand::None,
            Strand::ImportantButUnknown => GFFStrand::None,
        }
    }
}

impl FromSql for Strand {
    fn column_result(value: ValueRef) -> FromSqlResult<Self> {
        let result = match value.as_str() {
            Ok("+") => Strand::Forward,
            Ok("-") => Strand::Reverse,
            Ok(".") => Strand::Unknown,
            Ok("?") => Strand::ImportantButUnknown,
            _ => panic!("Invalid entry in database"),
        };
        Ok(result)
    }
}

impl fmt::Display for Strand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let result = match self {
            Strand::Forward => "+",
            Strand::Reverse => "-",
            Strand::Unknown => ".",
            Strand::ImportantButUnknown => "?",
        };
        formatter.write_str(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formats() {
        assert_eq!(format!("{v}", v = Strand::Forward), "+");
        assert_eq!(format!("{v}", v = Strand::Reverse), "-");
        assert_eq!(format!("{v}", v = Strand::ImportantButUnknown), "?");
        assert_eq!(format!("{v}", v = Strand::Unknown), ".");
    }

    #[test]
    fn test_column_conversion() {
        assert_eq!(
            Strand::column_result(ValueRef::Text("+".as_bytes())).unwrap(),
            Strand::Forward
        );
        assert_eq!(
            Strand::column_result(ValueRef::Text("-".as_bytes())).unwrap(),
            Strand::Reverse
        );
        assert_eq!(
            Strand::column_result(ValueRef::Text(".".as_bytes())).unwrap(),
            Strand::Unknown
        );
        assert_eq!(
            Strand::column_result(ValueRef::Text("?".as_bytes())).unwrap(),
            Strand::ImportantButUnknown
        );
    }
}
