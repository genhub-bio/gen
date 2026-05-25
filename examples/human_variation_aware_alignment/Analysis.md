# Human Variation-Aware Alignment

The human reference genome is a linear consensus of many individuals. Natural variation
reduces alignment scores and can suppress reads in certain areas, affecting variant calling.
By encoding known variation sites into the reference as a graph, we can use a graph aligner
to improve both mapping accuracy and variant detection.

This workflow uses Genome in a Bottle (GIAB) data and the `vg` graph aligner. It was run on
a machine with 128 GB RAM, 32 cores (m5.8xlarge), and 2 TB disk.

See `analysis.ipynb` for a self-contained Python version using small synthetic data.

## Download data

Download GIAB benchmark variants and the hg38 reference (~1 GB total):

```sh
wget https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/NA12878_HG001/NISTv4.2.1/GRCh38/HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```

Extract chromosome 1 and filter the VCF to chr1:

```sh
samtools faidx hg38.fa.gz chr1 | bgzip -c - > chr1.fa.gz
samtools faidx chr1.fa.gz
gunzip -c HG001_GRCh38_1_22_v4.2.1_benchmark.vcf.gz | awk '/^#/ || /^chr1\t/' | bgzip -c > chr1.vcf.bgz
```

## Build the variant graph

```sh
gen init
gen --db hg38.db import --name hg38 --fasta ./chr1.fa.gz --shallow
gen --db hg38.db update --name hg38 --vcf chr1.vcf.bgz
gen --db hg38.db export --name hg38 --gfa hg38.gfa
```

## Download vg and align reads

```sh
wget https://github.com/vgteam/vg/releases/download/v1.60.0/vg
chmod +x ./vg
```

Download GIAB HiSeq reads for NA12878 (multiple files, ~50 GB total):

```sh
wget -O R1.fq.gz <giab-r1-urls...>
wget -O R2.fq.gz <giab-r2-urls...>
```

Build graph indices and align:

```sh
./vg mod -X 32 hg38.gfa > hg38.mod.gfa
./vg autoindex -p index -w map -g hg38.mod.gfa
./vg map -x index.xg -g index.gcsa -f R1.fq.gz -f R2.fq.gz > align.gam
./vg gamsort -p align.gam > align.sorted.gam
./vg index -l align.sorted.gam
```

## Call variants (known sites)

```sh
./vg pack -e -x index.xg -g align.sorted.gam -o aln.pack
./vg call index.xg -k aln.pack > align.vcf
```

Example output — a known GIAB variant recovered from graph-aligned reads:

```
1-chr1  783006  >126167>126169  A  G  220.1  PASS  ...  0/1:38:24,14:...
1-chr1  783175  >126174>126176  T  C  24.9   PASS  ...  0/1:27:23,4:...
```

## Call novel variants

To call variants not in the input graph, augment the graph with soft-clipped read support:

```sh
vg convert index.xg > index.pg
vg augment index.pg align.sorted.gam -A aug.gam > augment.vg
vg index augment.vg -x augment.xg
vg pack -x augment.xg -g aug.gam -o aln.aug.pack
vg call augment.xg -k aln.aug.pack > align.aug.vcf
```

The augmented VCF contains both the known variants encoded in the graph and novel variants
discovered from read alignment discrepancies.

Example novel variant output:

```
1-chr1  10008  >20252164>9511549  AACCCTAACCCT...  CACCCTCCCATC...  30308.1  PASS  ...
```
