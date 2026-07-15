use std::fmt;

#[cfg(feature = "gff")]
use noodles::gff::feature::record::Strand as GFFStrand;
#[cfg(feature = "sqlite")]
use rusqlite::{
    ToSql,
    types::{FromSql, FromSqlResult, ToSqlOutput, Value, ValueRef},
};
use serde::{Deserialize, Serialize};

use crate::{errors::StrandError, gen_core_capnp};

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

    pub fn complement(self) -> Strand {
        match self {
            Strand::Forward => Strand::Reverse,
            Strand::Reverse => Strand::Forward,
            Strand::Unknown => Strand::Unknown,
            Strand::ImportantButUnknown => Strand::ImportantButUnknown,
        }
    }
}

// example https://docs.rs/rusqlite/latest/rusqlite/types/index.html
#[cfg(feature = "sqlite")]
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

#[cfg(feature = "sqlite")]
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

#[cfg(feature = "gff")]
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

#[cfg(feature = "gff")]
impl From<Strand> for GFFStrand {
    fn from(value: Strand) -> GFFStrand {
        match value {
            Strand::Forward => GFFStrand::Forward,
            Strand::Unknown => GFFStrand::Unknown,
            Strand::Reverse => GFFStrand::Reverse,
            Strand::ImportantButUnknown => GFFStrand::None,
        }
    }
}

#[cfg(feature = "sqlite")]
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

impl TryFrom<&str> for Strand {
    type Error = StrandError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "+" => Ok(Self::Forward),
            "-" => Ok(Self::Reverse),
            "." => Ok(Self::Unknown),
            "?" => Ok(Self::ImportantButUnknown),
            v => Err(StrandError::InvalidStrand(format!(
                "Invalid strand entry: {v}"
            ))),
        }
    }
}

impl From<Strand> for gen_core_capnp::Strand {
    fn from(value: Strand) -> gen_core_capnp::Strand {
        match value {
            Strand::Forward => gen_core_capnp::Strand::Forward,
            Strand::Unknown => gen_core_capnp::Strand::Unknown,
            Strand::Reverse => gen_core_capnp::Strand::Reverse,
            Strand::ImportantButUnknown => gen_core_capnp::Strand::ImportantButUnknown,
        }
    }
}

impl From<gen_core_capnp::Strand> for Strand {
    fn from(value: gen_core_capnp::Strand) -> Strand {
        match value {
            gen_core_capnp::Strand::Forward => Strand::Forward,
            gen_core_capnp::Strand::Unknown => Strand::Unknown,
            gen_core_capnp::Strand::Reverse => Strand::Reverse,
            gen_core_capnp::Strand::ImportantButUnknown => Strand::ImportantButUnknown,
        }
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

    #[cfg(feature = "sqlite")]
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

    #[test]
    fn test_complement() {
        assert_eq!(Strand::Forward.complement(), Strand::Reverse);
        assert_eq!(Strand::Reverse.complement(), Strand::Forward);
        assert_eq!(Strand::Unknown.complement(), Strand::Unknown);
        assert_eq!(
            Strand::ImportantButUnknown.complement(),
            Strand::ImportantButUnknown
        );
    }
}
