library(genr)
library(GenomicRanges)
library(BSgenome.Hsapiens.UCSC.hg38)

# Combinatorial library from genomic coordinates.
#
# Parts are defined as GRanges slices of an existing reference genome.
# The seq_containers argument maps each GRanges to its source genome by
# matching the 'genome' field in the GRanges Seqinfo slot against the
# assembly metadata of each container.
#
# Note: this approach extracts flat sequences via getSeq() under the hood.
# A future iteration will instead store coordinate-based Node references,
# reusing sequences already in the database rather than copying strings.

tmp <- tempfile(prefix = "gen-granges-library-")
dir.create(tmp)
setwd(tmp)
init()

repo <- Repository()

# Define parts as genomic coordinates.
# names() on the GRanges become part labels within each column.
promoters <- GRanges(
  seqnames = "chr1",
  ranges   = IRanges(
    start = c(1000, 2000, 3000),
    end   = c(1035, 2017, 3021)
  ),
  strand   = "+",
  seqinfo  = Seqinfo(seqnames = "chr1", genome = "hg38")
)
names(promoters) <- c("pA", "pB", "pC")

rbs_variants <- GRanges(
  seqnames = "chr1",
  ranges   = IRanges(
    start = c(5000, 6000),
    end   = c(5012, 6008)
  ),
  strand   = "+",
  seqinfo  = Seqinfo(seqnames = "chr1", genome = "hg38")
)
names(rbs_variants) <- c("rbs_strong", "rbs_weak")

# Pass the genome container alongside the GRanges columns.
# import_library matches each GRanges to the right container via the
# 'genome' field: GRanges Seqinfo genome == container assembly genome.
library_sg <- repo$import_library(
  library_name    = "genomic-parts-library",
  parts_list      = list(promoters, rbs_variants),
  seq_containers  = list(BSgenome.Hsapiens.UCSC.hg38),
  sample          = "design-v1",
  collection_name = "genomic-library"
)

plot(library_sg)
