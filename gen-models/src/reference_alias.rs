use std::collections::HashMap;

use anyhow::Result;
use rusqlite::Row;

use crate::{db::GraphConnection, traits::*};

#[derive(Debug, Clone)]
pub struct ReferenceAlias {
    pub reference_name: String,
    pub refseq_accession_id: String,
    pub genbank_accession_id: String,
}

impl Query for ReferenceAlias {
    type Model = ReferenceAlias;

    const TABLE_NAME: &'static str = "reference_aliases";

    fn process_row(row: &Row) -> Self::Model {
        ReferenceAlias {
            reference_name: row.get(0).unwrap(),
            refseq_accession_id: row.get(1).unwrap(),
            genbank_accession_id: row.get(2).unwrap(),
        }
    }
}

impl ReferenceAlias {
    pub fn create(
        conn: &GraphConnection,
        reference_name: &str,
        refseq_accession_id: &str,
        genbank_accession_id: &str,
    ) -> Result<()> {
        conn.execute(
            "INSERT INTO reference_aliases (reference_name, refseq_accession_id, genbank_accession_id) VALUES (?1, ?2, ?3)",
            rusqlite::params![reference_name, refseq_accession_id, genbank_accession_id],
        )?;
        Ok(())
    }

    pub fn load_all(conn: &GraphConnection) -> Result<HashMap<String, String>> {
        let mut stmt = conn.prepare("SELECT reference_name, refseq_accession_id, genbank_accession_id FROM reference_aliases")?;
        let reference_alias_iter = stmt.query_map([], |row| {
            Ok(ReferenceAlias {
                reference_name: row.get(0)?,
                refseq_accession_id: row.get(1)?,
                genbank_accession_id: row.get(2)?,
            })
        })?;

        let mut reference_aliases = HashMap::new();
        for reference_alias in reference_alias_iter {
            let reference_alias = reference_alias?;
            reference_aliases.insert(
                reference_alias.genbank_accession_id.clone(),
                reference_alias.refseq_accession_id.clone(),
            );
        }
        Ok(reference_aliases)
    }
}
