use std::path::Path;

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
    Gff3,
    Bed,
    Tabix,
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
            FileTypes::Gff3 => "gff3".into(),
            FileTypes::Bed => "bed".into(),
            FileTypes::Tabix => "tabix".into(),
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
            FileTypes::Gff3 => "gff3",
            FileTypes::Bed => "bed",
            FileTypes::Tabix => "tabix",
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
            Ok("gff3") => FileTypes::Gff3,
            Ok("bed") => FileTypes::Bed,
            Ok("tabix") => FileTypes::Tabix,
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
            FileTypes::Gff3 => gen_models_capnp::FileType::Gff3,
            FileTypes::Bed => gen_models_capnp::FileType::Bed,
            FileTypes::Tabix => gen_models_capnp::FileType::Tabix,
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
            gen_models_capnp::FileType::Gff3 => FileTypes::Gff3,
            gen_models_capnp::FileType::Bed => FileTypes::Bed,
            gen_models_capnp::FileType::Tabix => FileTypes::Tabix,
            gen_models_capnp::FileType::None => FileTypes::None,
        }
    }
}

impl FileTypes {
    pub fn infer_from_path(path: impl AsRef<Path>) -> Self {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase());

        match extension.as_deref() {
            Some("gb") | Some("gbk") | Some("genbank") => FileTypes::GenBank,
            Some("fa") | Some("fasta") | Some("fna") => FileTypes::Fasta,
            Some("gfa") => FileTypes::GFA,
            Some("gaf") => FileTypes::GAF,
            Some("vcf") => FileTypes::VCF,
            Some("csv") => FileTypes::CSV,
            Some("gff") | Some("gff3") => FileTypes::Gff3,
            Some("bed") => FileTypes::Bed,
            Some("tbi") => FileTypes::Tabix,
            _ => FileTypes::None,
        }
    }

    pub fn suffix(file_type: FileTypes) -> String {
        let result = match file_type {
            FileTypes::GenBank => "gb",
            FileTypes::Fasta => "fa",
            FileTypes::GFA => "gfa",
            FileTypes::VCF => "vcf",
            FileTypes::Changeset => "cs",
            FileTypes::CSV => "csv",
            FileTypes::GAF => "gaf",
            FileTypes::Gff3 => "gff3",
            FileTypes::Bed => "bed",
            FileTypes::Tabix => "tbi",
            FileTypes::None => "none",
        };

        result.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::FileTypes;

    #[test]
    fn infers_file_type_from_path_extension() {
        assert_eq!(FileTypes::infer_from_path("sample.gb"), FileTypes::GenBank);
        assert_eq!(FileTypes::infer_from_path("sample.GBK"), FileTypes::GenBank);
        assert_eq!(FileTypes::infer_from_path("sample.fa"), FileTypes::Fasta);
        assert_eq!(FileTypes::infer_from_path("sample.gfa"), FileTypes::GFA);
        assert_eq!(FileTypes::infer_from_path("sample.gaf"), FileTypes::GAF);
        assert_eq!(FileTypes::infer_from_path("sample.vcf"), FileTypes::VCF);
        assert_eq!(FileTypes::infer_from_path("sample.csv"), FileTypes::CSV);
        assert_eq!(FileTypes::infer_from_path("sample.gff3"), FileTypes::Gff3);
        assert_eq!(FileTypes::infer_from_path("sample.bed"), FileTypes::Bed);
        assert_eq!(FileTypes::infer_from_path("sample.tbi"), FileTypes::Tabix);
        assert_eq!(
            FileTypes::infer_from_path("sample.unknown"),
            FileTypes::None
        );
        assert_eq!(FileTypes::infer_from_path("sample"), FileTypes::None);
    }
}
