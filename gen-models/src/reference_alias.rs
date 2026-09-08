use std::collections::{HashMap, HashSet};

use rusqlite::Result;
use thiserror::Error;

use crate::{ModelSelect, db::GraphConnection};

#[derive(Debug, Clone, ModelSelect)]
#[model_select(table = "reference_aliases")]
pub struct ReferenceAlias {
    pub reference_name: String,
    pub refseq_accession_id: Option<String>,
    pub genbank_accession_id: Option<String>,
    pub ucsc_id: Option<String>,
    pub ensembl_id: Option<String>,
    pub custom_id: Option<String>,
    pub chromosome: Option<i64>,
}

#[derive(Debug, Error)]
pub enum ReferenceAliasError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
}

impl ReferenceAlias {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        conn: &GraphConnection,
        reference_name: &str,
        refseq_accession_id: Option<String>,
        genbank_accession_id: Option<String>,
        ucsc_id: Option<String>,
        ensembl_id: Option<String>,
        custom_id: Option<String>,
        chromosome: Option<i64>,
    ) -> rusqlite::Result<ReferenceAlias, ReferenceAliasError> {
        conn.execute(
            "INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, custom_id, chromosome) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![reference_name, refseq_accession_id, genbank_accession_id, ucsc_id, ensembl_id, custom_id, chromosome],
        )?;

        Ok(ReferenceAlias {
            reference_name: reference_name.to_string(),
            refseq_accession_id,
            genbank_accession_id,
            ucsc_id,
            ensembl_id,
            custom_id,
            chromosome,
        })
    }

    fn chromosome_aliases(alias_id: &str) -> Vec<String> {
        vec![
            format!("chr{}", alias_id),
            format!("Chr{}", alias_id),
            format!("chrom{}", alias_id),
            format!("Chrom{}", alias_id),
            format!("chromosome{}", alias_id),
            format!("Chromosome{}", alias_id),
        ]
    }

    fn compute_aliases(reference_alias: ReferenceAlias) -> HashSet<String> {
        let mut aliases = HashSet::new();
        if let Some(refseq_id) = reference_alias.refseq_accession_id {
            aliases.insert(refseq_id.clone());
            aliases.insert(format!("ref|{}|", refseq_id));
            if refseq_id.contains('.') {
                let refseq_without_version = refseq_id.split('.').next().unwrap();
                aliases.insert(refseq_without_version.to_string());
                aliases.insert(format!("ref|{}|", refseq_without_version));
            }
        }
        if let Some(genbank_id) = reference_alias.genbank_accession_id.clone() {
            aliases.insert(genbank_id.clone());
            if genbank_id.contains('.') {
                let genbank_without_version = genbank_id.split('.').next().unwrap();
                aliases.insert(genbank_without_version.to_string());
            }
        }
        if let Some(ucsc_id) = reference_alias.ucsc_id.clone() {
            aliases.insert(ucsc_id.clone());
        }
        if let Some(ensembl_id) = reference_alias.ensembl_id.clone() {
            aliases.insert(ensembl_id.clone());
            aliases.extend(ReferenceAlias::chromosome_aliases(&ensembl_id));
        }
        if let Some(custom_id) = reference_alias.custom_id {
            aliases.insert(custom_id.clone());
            aliases.extend(ReferenceAlias::chromosome_aliases(&custom_id));
        }
        if let Some(chromosome) = reference_alias.chromosome {
            aliases.insert(chromosome.to_string());
            aliases.extend(ReferenceAlias::chromosome_aliases(&chromosome.to_string()));
        }
        aliases
    }

    pub fn get_references_by_alias(
        conn: &GraphConnection,
        references: Vec<String>,
        history_ref: Option<&str>,
    ) -> Result<HashMap<String, String>, ReferenceAliasError> {
        let mut references_by_alias = HashMap::new();
        let select = ReferenceAlias::select(conn).with_ref(history_ref);
        let reference_aliases = select.load().expect("should load reference aliases");
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
            Some("REFSEQ123".to_string()),
            Some("GENBANK123".to_string()),
            Some("UCSC123".to_string()),
            Some("ENSEMBL123".to_string()),
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
        assert_eq!(new_entry.refseq_accession_id, Some("REFSEQ123".to_string()));
    }

    #[test]
    fn test_prepopulated_aliases() {
        let conn = &mut get_connection(None).unwrap();
        let reference_aliases = ReferenceAlias::all(conn).expect("should load reference aliases");
        assert_eq!(reference_aliases.len(), 107);
        let first_e_coli_reference = reference_aliases
            .iter()
            .find(|alias| alias.genbank_accession_id == Some("U00096.3".to_string()))
            .unwrap();
        let aliases = ReferenceAlias::compute_aliases(first_e_coli_reference.clone());
        assert!(aliases.contains("NC_000913.3"));
        assert!(aliases.contains("NC_000913"));
        assert!(aliases.contains("ref|NC_000913|"));
        assert!(aliases.contains("U00096.3"));
        assert!(aliases.contains("U00096"));

        let first_yeast_reference = reference_aliases
            .iter()
            .find(|alias| alias.genbank_accession_id == Some("BK006935.2".to_string()))
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
