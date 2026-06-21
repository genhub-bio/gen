use anyhow::{Error, Result};
use gen_core::{Strand, region::Region};
use gen_models::{db::DbContext, region::resolve};

use crate::{
    commands::get_default_collection,
    graphs::translation::{
        CodonTable, TranslationOperationError, TranslationParams, translate_annotation,
        translate_from_path, with_translation_operation,
    },
};

pub struct DeriveTranslationArgs {
    pub collection: Option<String>,
    pub sample: String,
    pub region: String,
    pub name: Option<String>,
    pub strand: Option<String>,
    pub frame: u8,
    pub codon_table: u8,
}

pub fn derive_translation_operation(
    db_context: &DbContext,
    args: DeriveTranslationArgs,
) -> Result<(), Error> {
    let operation_conn = db_context.operations().conn();
    let graph_conn = db_context.graph().conn();

    let collection_name = match args.collection {
        Some(c) => c,
        None => get_default_collection(operation_conn),
    };

    let resolved_strand = match args.strand.as_deref() {
        None => None,
        Some("forward") => Some(Strand::Forward),
        Some("reverse") => Some(Strand::Reverse),
        Some(s) => {
            return Err(Error::msg(format!(
                "Unknown strand '{s}'; use 'forward' or 'reverse'"
            )));
        }
    };

    let table = CodonTable::ncbi(args.codon_table)
        .ok_or_else(|| Error::msg(format!("Invalid codon table id {}", args.codon_table)))?;

    let mut tr_params = TranslationParams::new(&collection_name)
        .initial_frame(args.frame)
        .map_err(|e| Error::msg(e.to_string()))?
        .codon_table(table);
    if let Some(s) = resolved_strand {
        tr_params = tr_params.strand(s);
    }
    if let Some(name) = args.name.as_deref() {
        tr_params = tr_params.name(name);
    }

    let region_str = &args.region;
    let region = Region::parse(region_str)
        .map_err(|e| Error::msg(format!("Invalid region '{region_str}': {e}")))?;

    let resolved = resolve(&region, graph_conn, &collection_name, &args.sample)
        .map_err(|e| Error::msg(e.to_string()))?;

    let bg_id = resolved.block_group.id;

    // When the region names an annotation without coordinates, use translate_annotation
    // to take its entry point and strand from the annotation itself. When coordinates are
    // present or the region is a block group / path, translate from the resolved start
    // coordinate instead. Either way translation reads forward to its own first in-frame
    // stop codon; neither path is bounded by a declared end coordinate.
    let protein_bg = if resolved.annotation.is_some()
        && resolved.start == resolved.anchor_start
        && resolved.end == resolved.anchor_end
    {
        with_translation_operation(db_context, region_str, || {
            translate_annotation(
                graph_conn,
                resolved.annotation.as_ref().unwrap(),
                Some(&bg_id),
                tr_params,
            )
        })
        .map_err(op_err)?
    } else {
        with_translation_operation(db_context, region_str, || {
            translate_from_path(graph_conn, &bg_id, resolved.start, tr_params)
        })
        .map_err(op_err)?
    };
    let label = region_str.clone();

    println!(
        "Translated '{}' → protein sequence graph '{}'.",
        label, protein_bg.name
    );

    Ok(())
}

fn op_err(e: TranslationOperationError) -> Error {
    Error::msg(e.to_string())
}
