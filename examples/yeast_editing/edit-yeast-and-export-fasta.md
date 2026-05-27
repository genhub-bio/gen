# Yeast Genome Editing

Demonstrates editing a yeast genome sequence and exporting the result. We replace a short
subsequence of chromosome VI (`NC_001138`) with a synthetic cassette insert (`ATCGATCG`).

## Setup

Download and extract the S288C reference genome (~16 MB):

```sh
wget http://sgd-archive.yeastgenome.org/sequence/S288C_reference/genome_releases/S288C_reference_genome_R64-1-1_20110203.tgz
tar -xzvf S288C_reference_genome_R64-1-1_20110203.tgz
```

```
x S288C_reference_genome_R64-1-1_20110203/
x S288C_reference_genome_R64-1-1_20110203/S288C_reference_sequence_R64-1-1_20110203.fsa
x S288C_reference_genome_R64-1-1_20110203/gene_association_R64-1-1_20110205.sgd
x S288C_reference_genome_R64-1-1_20110203/saccharomyces_cerevisiae_R64-1-1_20110208.gff
x S288C_reference_genome_R64-1-1_20110203/other_features_genomic_R64-1-1_20110203.fasta
...
```

Initialize a repository:

```sh
gen init
gen defaults --database yeast.db --collection genome
```

```
Gen repository initialized.
Default database set to yeast.db
Default collection set to genome
```

## Import the reference genome

```sh
gen import --fasta S288C_reference_genome_R64-1-1_20110203/S288C_reference_sequence_R64-1-1_20110203.fsa
```

```
Created it
```

## Apply an edit

Create a FASTA file containing the cassette sequence to insert:

```sh
echo ">foo" > cassette-edit.fa
echo "ATCGATCG" >> cassette-edit.fa
```

Apply the edit, replacing bases 3–5 of `NC_001138` with the cassette:

```sh
gen update --fasta cassette-edit.fa --new-sample edited-sample --start 3 --end 5 --region-name "ref|NC_001138|"
```

```
Updated with fasta file: cassette-edit.fa
```

## Export the edited genome

```sh
gen export --fasta edited-yeast-genome.fa --sample edited-sample
```

```
Exported to file edited-yeast-genome.fa
```

## Verify the edit

Reference sequence around `NC_001138`:

```sh
grep -A 3 NC_001138 S288C_reference_genome_R64-1-1_20110203/S288C_reference_sequence_R64-1-1_20110203.fsa
```

```
>ref|NC_001138| [org=Saccharomyces cerevisiae] [strain=S288C] [moltype=genomic] [chromosome=VI]
GATCTCGCAAGTGCATTCCTAGACTTAATTCATATCTGCTCCTCAACTGTCGATGATGCC
TGCTAAACTGCAGCTTGACGTACTGCGGACCCTGCAGTCCAGCGCTCGTCATGGAACGCA
...
```

Edited sequence — `ATCGATCG` inserted after the third base pair:

```sh
grep -A 3 NC_001138 edited-yeast-genome.fa
```

```
>ref|NC_001138|
GATATCGATCGCGCAAGTGCATTCCTAGACTTAATTCATATCTGCTCCTCAACTGTCGATGATGCC
TGCTAAACTGCAGCTTGACGTACTGCGGACCCTGCAGTCCAGCGCTCGTCATGGAACGCA
...
```
