# List and Show

Demonstrating convenience subcommands for listing samples, graphs, and retrieving sequences.

## Setup

Initialize a repository and set defaults:

```sh
gen init
gen defaults --database simple.db --collection simple
```

```
Gen repository initialized.
Default database set to simple.db
Default collection set to simple
```

Import a reference FASTA and apply variants from a VCF:

```sh
gen import --fasta simple.fa
gen update --vcf simple.vcf
```

```
Fasta imported.
```

## List samples

After applying the VCF, one block group is created per sample found in the VCF:

```sh
gen list-samples
```

```
unknown
G1
foo
```

## List graphs for a sample

```sh
gen list-graphs --sample foo
```

```
m123
```

## Get a sequence

Retrieve the full path sequence for a sample and graph:

```sh
gen get-sequence --sample foo --graph m123
```

```
ATCGATCGATCGATCGATCGGGAACACACAGAGA
```

Retrieve a subsequence by position (0-based, half-open interval):

```sh
gen get-sequence --sample unknown --graph m123 --start 20 --end 30
```

```
GGAACACACA
```
