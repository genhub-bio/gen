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

impl AsRef<str> for FileTypes {
    fn as_ref(&self) -> &str {
        self.storage_tag()
    }
}

impl ToSql for FileTypes {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(self.as_ref().into())
    }
}

impl From<FileTypes> for Value {
    fn from(value: FileTypes) -> Value {
        Value::Text(value.as_ref().to_string())
    }
}

impl FromSql for FileTypes {
    fn column_result(value: ValueRef) -> FromSqlResult<Self> {
        let tag = value.as_str()?;
        Ok(FileTypes::from_storage_tag(tag))
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
    pub fn from_storage_tag(tag: &str) -> Self {
        match tag {
            "gb" => FileTypes::GenBank,
            "fasta" => FileTypes::Fasta,
            "gfa" => FileTypes::GFA,
            "gaf" => FileTypes::GAF,
            "vcf" => FileTypes::VCF,
            "changeset" => FileTypes::Changeset,
            "csv" => FileTypes::CSV,
            "gff3" => FileTypes::Gff3,
            "bed" => FileTypes::Bed,
            "tabix" => FileTypes::Tabix,
            "none" => FileTypes::None,
            _ => FileTypes::None,
        }
    }

    fn storage_tag(self) -> &'static str {
        match self {
            FileTypes::GenBank => "gb",
            FileTypes::Fasta => "fasta",
            FileTypes::GFA => "gfa",
            FileTypes::GAF => "gaf",
            FileTypes::VCF => "vcf",
            FileTypes::Changeset => "changeset",
            FileTypes::CSV => "csv",
            FileTypes::Gff3 => "gff3",
            FileTypes::Bed => "bed",
            FileTypes::Tabix => "tabix",
            FileTypes::None => "none",
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.storage_tag()
    }

    pub fn infer_from_path(path: impl AsRef<Path>) -> Self {
        let extensions = path
            .as_ref()
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .split('.')
            .skip(1)
            .collect::<Vec<_>>();

        for extension in extensions.iter().rev() {
            let file_type = Self::infer_from_extension(extension);
            if file_type != FileTypes::None {
                return file_type;
            }
        }

        FileTypes::None
    }

    pub fn infer_from_extension(extension: &str) -> Self {
        match extension.to_ascii_lowercase().as_str() {
            "gb" | "gbk" | "genbank" => FileTypes::GenBank,
            "fa" | "fasta" | "fna" => FileTypes::Fasta,
            "gfa" => FileTypes::GFA,
            "gaf" => FileTypes::GAF,
            "vcf" => FileTypes::VCF,
            "csv" => FileTypes::CSV,
            "gff" | "gff3" => FileTypes::Gff3,
            "bed" => FileTypes::Bed,
            "tbi" => FileTypes::Tabix,
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
    fn test_file_type_as_ref_returns_canonical_storage_tag() {
        assert_eq!(<FileTypes as AsRef<str>>::as_ref(&FileTypes::GenBank), "gb");
        assert_eq!(
            <FileTypes as AsRef<str>>::as_ref(&FileTypes::Tabix),
            "tabix"
        );
        assert_eq!(<FileTypes as AsRef<str>>::as_ref(&FileTypes::None), "none");
    }

    #[test]
    fn test_unknown_storage_tag_maps_to_none() {
        let parsed = <FileTypes as rusqlite::types::FromSql>::column_result(
            rusqlite::types::ValueRef::Text(b"unexpected"),
        )
        .expect("should parse unknown file type as none");

        assert_eq!(parsed, FileTypes::None);
    }

    #[test]
    fn infers_file_type_from_path_extension() {
        assert_eq!(FileTypes::infer_from_path("sample.gb"), FileTypes::GenBank);
        assert_eq!(FileTypes::infer_from_path("sample.GBK"), FileTypes::GenBank);
        assert_eq!(FileTypes::infer_from_path("sample.fa"), FileTypes::Fasta);
        assert_eq!(FileTypes::infer_from_path("sample.fa.gz"), FileTypes::Fasta);
        assert_eq!(
            FileTypes::infer_from_path("sample.fa.bgz"),
            FileTypes::Fasta
        );
        assert_eq!(
            FileTypes::infer_from_path("https://example.com/assets/sample.fa.gz"),
            FileTypes::Fasta
        );
        assert_eq!(
            FileTypes::infer_from_path("https://example.com/assets/sample.fa.gz?download=1"),
            FileTypes::Fasta
        );
        assert_eq!(
            FileTypes::infer_from_path("https://example.com/assets/sample.fa.gz#reference"),
            FileTypes::Fasta
        );
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
