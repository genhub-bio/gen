# Annotated genomes with variants

These files are copied unchanged from the corresponding examples and exercise annotation
placement across graph nodes split by SNPs and deletions.

- `hbb_reference_centering/`: a 6,000-base GRCh38 chromosome 11 region around HBB,
  eight Ensembl gene/transcript/exon/CDS records, and phased variants for four
  1000 Genomes samples. This is a human genome excerpt, not a complete genome.
- `sarscov2_reference_centering/`: the complete 29,903-base Wuhan-Hu-1 reference
  NC_045512.2, NCBI GFF3 annotations, and variants for Alpha, Delta, and Omicron BA.1.

The six data files total 56,652 bytes. Sequence names match across each FASTA/GFF3/VCF
set. Import the FASTA as the reference sample before applying its VCF.

Source and preparation details are in
`examples/hbb_reference_centering/prepare_data.py` and
`examples/sarscov2_reference_centering/build_data.py`; the GFF3 headers also retain
source metadata. The original example notebooks demonstrate the import workflows.
