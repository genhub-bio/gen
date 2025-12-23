use rusqlite::{
    ToSql,
    types::{FromSql, FromSqlResult, ToSqlOutput, Value, ValueRef},
};
use serde::{Deserialize, Serialize};

use crate::gen_models_capnp;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Deserialize, Serialize)]
pub enum FileTypes {
    GenBank,
    Fasta,
    GFA,
    GAF,
    VCF,
    Changeset,
    CSV,
    None,
}

impl ToSql for FileTypes {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        let result = match self {
            FileTypes::GenBank => "gb".into(),
            FileTypes::Fasta => "fasta".into(),
            FileTypes::GFA => "gfa".into(),
            FileTypes::VCF => "vcf".into(),
            FileTypes::Changeset => "changeset".into(),
            FileTypes::CSV => "csv".into(),
            FileTypes::GAF => "gaf".into(),
            FileTypes::None => "none".into(),
        };
        Ok(result)
    }
}

impl From<FileTypes> for Value {
    fn from(value: FileTypes) -> Value {
        let result = match value {
            FileTypes::GenBank => "gb",
            FileTypes::Fasta => "fasta",
            FileTypes::GFA => "gfa",
            FileTypes::VCF => "vcf",
            FileTypes::Changeset => "changeset",
            FileTypes::CSV => "csv",
            FileTypes::GAF => "gaf",
            FileTypes::None => "none",
        };
        Value::Text(result.to_string())
    }
}

impl FromSql for FileTypes {
    fn column_result(value: ValueRef) -> FromSqlResult<Self> {
        let result = match value.as_str() {
            Ok("gb") => FileTypes::GenBank,
            Ok("fasta") => FileTypes::Fasta,
            Ok("gfa") => FileTypes::GFA,
            Ok("vcf") => FileTypes::VCF,
            Ok("changeset") => FileTypes::Changeset,
            Ok("csv") => FileTypes::CSV,
            Ok("gaf") => FileTypes::GAF,
            Ok("none") => FileTypes::None,
            _ => panic!("Invalid entry in database"),
        };
        Ok(result)
    }
}
impl From<FileTypes> for gen_models_capnp::FileType {
    fn from(value: FileTypes) -> gen_models_capnp::FileType {
        match value {
            FileTypes::GenBank => gen_models_capnp::FileType::GenBank,
            FileTypes::Fasta => gen_models_capnp::FileType::Fasta,
            FileTypes::GFA => gen_models_capnp::FileType::Gfa,
            FileTypes::GAF => gen_models_capnp::FileType::Gaf,
            FileTypes::VCF => gen_models_capnp::FileType::Vcf,
            FileTypes::Changeset => gen_models_capnp::FileType::Changeset,
            FileTypes::CSV => gen_models_capnp::FileType::Csv,
            FileTypes::None => gen_models_capnp::FileType::None,
        }
    }
}

impl From<gen_models_capnp::FileType> for FileTypes {
    fn from(value: gen_models_capnp::FileType) -> FileTypes {
        match value {
            gen_models_capnp::FileType::GenBank => FileTypes::GenBank,
            gen_models_capnp::FileType::Fasta => FileTypes::Fasta,
            gen_models_capnp::FileType::Gfa => FileTypes::GFA,
            gen_models_capnp::FileType::Gaf => FileTypes::GAF,
            gen_models_capnp::FileType::Vcf => FileTypes::VCF,
            gen_models_capnp::FileType::Changeset => FileTypes::Changeset,
            gen_models_capnp::FileType::Csv => FileTypes::CSV,
            gen_models_capnp::FileType::None => FileTypes::None,
        }
    }
}

impl FileTypes {
    pub fn suffix(file_type: FileTypes) -> String {
        let result = match file_type {
            FileTypes::GenBank => "gb",
            FileTypes::Fasta => "fa",
            FileTypes::GFA => "gfa",
            FileTypes::VCF => "vcf",
            FileTypes::Changeset => "cs",
            FileTypes::CSV => "csv",
            FileTypes::GAF => "gaf",
            FileTypes::None => "none",
        };

        result.to_string()
    }
}
