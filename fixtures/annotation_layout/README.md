# Annotated genomes with variants

Test inputs for annotation placement across graph nodes split by SNPs and deletions.

- `hbb_region/`: a 6,000-base GRCh38 chromosome 11 region around HBB (sequence and
  annotation from the Ensembl REST API, `11:5224001-5230000`, relabelled `HBB_region`),
  eight gene, transcript, exon and CDS records, and phased variants for four
  1000 Genomes samples. This is a human genome excerpt, not a complete genome.
- `sarscov2_genome/`: the complete 29,903-base Wuhan-Hu-1 reference NC_045512.2 with its
  NCBI GFF3 annotations, and a simplified variant file for Alpha, Delta, and Omicron BA.1
  derived from lineage-defining mutations. The variants are a test input, not observed
  genomes.

The six data files total 56,652 bytes. Sequence names match across each FASTA/GFF3/VCF
set. Import the FASTA as the reference sample before applying its VCF.

Rust snapshot tests import each FASTA and VCF, translate the GFF3 onto HG00096 (HBB)
or Delta (SARS-CoV-2), and render the compact view around annotated variants using
the stored DNA sequences. Their ASCII snapshots are in `src/views/snapshots/` as
`annotation_hbb_vcf` and `annotation_sarscov2_vcf`.

Run both with `cargo test -p gen --lib annotation_snapshots`. To intentionally update
the snapshots, run the same command with `INSTA_UPDATE=always` and review the diff.
