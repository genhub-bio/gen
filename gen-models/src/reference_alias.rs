use std::collections::{HashMap, HashSet};

use rusqlite::{Result, Row};
use thiserror::Error;

use crate::{db::GraphConnection, traits::*};

#[derive(Debug, Clone)]
pub struct ReferenceAlias {
    pub reference_name: String,
    pub refseq_accession_id: String,
    pub genbank_accession_id: String,
    pub ucsc_id: String,
    pub ensembl_id: String,
    pub custom_id: Option<String>,
    pub chromosome: Option<i64>,
}

#[derive(Debug, Error)]
pub enum ReferenceAliasError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl Query for ReferenceAlias {
    type Model = ReferenceAlias;

    const TABLE_NAME: &'static str = "reference_aliases";

    fn process_row(row: &Row) -> Self::Model {
        ReferenceAlias {
            reference_name: row.get(0).unwrap(),
            refseq_accession_id: row.get(1).unwrap(),
            genbank_accession_id: row.get(2).unwrap(),
            ucsc_id: row.get(3).unwrap(),
            ensembl_id: row.get(4).unwrap(),
            custom_id: row.get(5).unwrap(),
            chromosome: row.get(6).unwrap(),
        }
    }
}

impl ReferenceAlias {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        conn: &GraphConnection,
        reference_name: &str,
        refseq_accession_id: &str,
        genbank_accession_id: &str,
        ucsc_id: &str,
        ensembl_id: &str,
        custom_id: Option<String>,
        chromosome: Option<i64>,
    ) -> rusqlite::Result<ReferenceAlias, ReferenceAliasError> {
        conn.execute(
            "INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, custom_id, chromosome) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, custom_id, chromosome],
        )?;

        Ok(ReferenceAlias {
            reference_name: reference_name.to_string(),
            refseq_accession_id: refseq_accession_id.to_string(),
            genbank_accession_id: genbank_accession_id.to_string(),
            ucsc_id: ucsc_id.to_string(),
            ensembl_id: ensembl_id.to_string(),
            custom_id,
            chromosome,
        })
    }

    fn compute_aliases(reference_alias: ReferenceAlias) -> HashSet<String> {
        let mut aliases = HashSet::new();
        aliases.insert(reference_alias.refseq_accession_id.clone());
        aliases.insert(format!("ref|{}|", reference_alias.refseq_accession_id));
        if reference_alias.refseq_accession_id.contains('.') {
            let refseq_without_version = reference_alias
                .refseq_accession_id
                .split('.')
                .next()
                .unwrap();
            aliases.insert(refseq_without_version.to_string());
            aliases.insert(format!("ref|{}|", refseq_without_version));
        }
        aliases.insert(reference_alias.genbank_accession_id.clone());
        if reference_alias.genbank_accession_id.contains('.') {
            let genbank_without_version = reference_alias
                .genbank_accession_id
                .split('.')
                .next()
                .unwrap();
            aliases.insert(genbank_without_version.to_string());
        }
        aliases.insert(reference_alias.ucsc_id);
        aliases.insert(reference_alias.ensembl_id.clone());
        aliases.insert(format!("chr{}", reference_alias.ensembl_id));
        aliases.insert(format!("Chr{}", reference_alias.ensembl_id));
        aliases.insert(format!("chrom{}", reference_alias.ensembl_id));
        aliases.insert(format!("Chrom{}", reference_alias.ensembl_id));
        aliases.insert(format!("chromosome{}", reference_alias.ensembl_id));
        aliases.insert(format!("Chromosome{}", reference_alias.ensembl_id));
        if let Some(custom_id) = reference_alias.custom_id {
            aliases.insert(custom_id.clone());
            aliases.insert(format!("chr{}", custom_id));
            aliases.insert(format!("Chr{}", custom_id));
            aliases.insert(format!("chrom{}", custom_id));
            aliases.insert(format!("Chrom{}", custom_id));
            aliases.insert(format!("chr{}", custom_id));
            aliases.insert(format!("Chromosome{}", custom_id));
        }
        if let Some(chromosome) = reference_alias.chromosome {
            aliases.insert(chromosome.to_string());
            aliases.insert(format!("chr{}", chromosome));
            aliases.insert(format!("Chr{}", chromosome));
            aliases.insert(format!("chrom{}", chromosome));
            aliases.insert(format!("Chrom{}", chromosome));
            aliases.insert(format!("chromosome{}", chromosome));
            aliases.insert(format!("Chromosome{}", chromosome));
        }
        aliases
    }

    pub fn get_references_by_alias(
        conn: &GraphConnection,
        references: Vec<String>,
    ) -> Result<HashMap<String, String>, ReferenceAliasError> {
        let mut references_by_alias = HashMap::new();
        let reference_aliases = ReferenceAlias::all(conn);
        for reference_alias in reference_aliases {
            let aliases = ReferenceAlias::compute_aliases(reference_alias);
            for reference in &references {
                if aliases.contains(reference) {
                    for alias in &aliases {
                        references_by_alias.insert(alias.clone(), reference.to_string());
                    }
                }
            }
        }
        Ok(references_by_alias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::get_connection;

    #[test]
    fn test_create() {
        let conn = &mut get_connection(None).unwrap();
        ReferenceAlias::create(
            conn,
            "Test Reference",
            "REFSEQ123",
            "GENBANK123",
            "UCSC123",
            "ENSEMBL123",
            Some("CUSTOM123".to_string()),
            Some(1),
        )
        .unwrap();
        let new_entry = conn
	    .query_row(
		"SELECT reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, custom_id, chromosome FROM reference_aliases WHERE reference_name = ?1",
		rusqlite::params!["Test Reference"],
		|row| {
		    Ok(ReferenceAlias {
			reference_name: row.get(0)?,
			refseq_accession_id: row.get(1)?,
			genbank_accession_id: row.get(2)?,
			ucsc_id: row.get(3)?,
			ensembl_id: row.get(4)?,
			custom_id: row.get(5)?,
			chromosome: row.get(6)?,
		    })
		},
	    )
	    .unwrap();
        assert_eq!(new_entry.reference_name, "Test Reference");
        assert_eq!(new_entry.refseq_accession_id, "REFSEQ123");
    }

    #[test]
    fn test_prepopulated_aliases() {
        let conn = &mut get_connection(None).unwrap();
        let reference_aliases = ReferenceAlias::all(conn);
        assert_eq!(reference_aliases.len(), 107);
        let first_e_coli_reference = reference_aliases
            .iter()
            .find(|alias| alias.genbank_accession_id == "U00096.3")
            .unwrap();
        let aliases = ReferenceAlias::compute_aliases(first_e_coli_reference.clone());
        assert!(aliases.contains("NC_000913.3"));
        assert!(aliases.contains("NC_000913"));
        assert!(aliases.contains("ref|NC_000913|"));
        assert!(aliases.contains("U00096.3"));
        assert!(aliases.contains("U00096"));

        let first_yeast_reference = reference_aliases
            .iter()
            .find(|alias| alias.genbank_accession_id == "BK006935.2")
            .unwrap();
        let aliases = ReferenceAlias::compute_aliases(first_yeast_reference.clone());
        assert!(aliases.contains("BK006935.2"));
        assert!(aliases.contains("BK006935"));
        assert!(aliases.contains("NC_001133.9"));
        assert!(aliases.contains("NC_001133"));
        assert!(aliases.contains("ref|NC_001133|"));
        assert!(aliases.contains("chrI"));
        assert!(aliases.contains("chr1"));
    }
}
