library(genr)
library(Biostrings)

# Combinatorial expression cassette library.
#
# Layout (5' -> 3'):
#   [fixed upstream] - [promoter x3] - [RBS x3] - [CDS x2] - [fixed terminator]
#
# 1 x 3 x 3 x 2 x 1 = 18 unique combinations.

tmp <- tempfile(prefix = "gen-cassette-")
dir.create(tmp)
setwd(tmp)
init()

repo <- Repository()

# Single-member columns use a length-1 DNAStringSet.
# Plain char equivalent: c(upstream = "AATTCGGATCCAAGCTT")
upstream <- DNAStringSet(c(upstream = "AATTCGGATCCAAGCTT"))

# Multi-member columns: each name is a part label, each value its sequence.
# Plain char equivalent: c(pTrc = "TTGAC...", pT7 = "TAATA...", pLac = "AATTG...")
promoters <- DNAStringSet(c(
  pTrc  = "TTGACAATTAATCATCCGGCTCGTATAATGTGTGG",
  pT7   = "TAATACGACTCACTATA",
  pLac  = "AATTGTGAGCGGATAACAATT"
))

rbs_variants <- DNAStringSet(c(
  rbs_strong = "AAAGAGGAGAAA",
  rbs_medium = "AAGAGGAG",
  rbs_weak   = "AGGAG"
))

# CDS sequences can also be loaded from a FASTA file:
#   cds_variants <- readDNAStringSet("cds_parts.fa")
cds_variants <- DNAStringSet(c(
  gfp = "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGG",
  rfp = "ATGGCTTCCTCCGAAGACGTTATCAAAGAGTTCATGCGCTTCAAGGTGCGCATGGAGGGCTCCGTGAAC"
))

terminator <- DNAStringSet(c(terminator_T1 = "GCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGC"))

repo$import_library(
  library_name    = "expression-cassette",
  parts_list      = list(upstream, promoters, rbs_variants, cds_variants, terminator),
  sample          = "design-v1",
  collection_name = "cassette-library"
)

bgs <- repo$get_block_groups_by_collection("cassette-library")
cassette_bg <- bgs[[1]]

repo$plot(cassette_bg)
