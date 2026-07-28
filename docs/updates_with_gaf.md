# Updates with alignments

Suppose we are working with a sample where we are unable to provide an unambiguous coordinate into a chromosome. This
is a common scenario in non-haploid organisms with tracked heterozygous changes (for example, suppose after each round
of engineering, observed sequencing variants are incorporated into the graph). For these scenarios, updates need to
be made with respect to node ids in the graph. This is a cumbersome process to identify nodes and determine positions, 
and is an area we are constantly trying to improve. One manner to circumvent this is to use alignments and the output
of graph aligners to localize changes. The following example walks through updating a sample where after each round
of engineering, background changes are recorded into the graph.

Our sample in this case is called `K562`, a human cell line. We wish to insert a gene into chromsome 22. To do this, we will
take the region flanking the left and right side of the insertion point, and use a graph aligner to identify the nodes
in our graph corresponding to this region. Then, the provided insert will be placed in between the bounds of that region.

Although most human genomes are present as a linear fasta file, for this example we use a portion of chr22 from Hg38, 
located in [fixtures/chr22_het.gfa](../fixtures/chr22_het.gfa). We start by importing this file to a database

```console
gen import --name k562 --gaf chr22_het.gfa
```

For this, we accept a csv file format which gen will autoformat into a fasta file for each update we wish to make.

example.csv
```
id,left,sequence,right
simple_insert,atgggagtataattttagatagtgaagatttctgtattcaaatgccacat,atcgatcgatcggagaga,acacagaaaaaggcaggcagagaaaataacaaggataaagacactgaagt
```

The id field is optional. If provided, an [accession](path_accessions.md) will be created with the id given.

NOTE: This is not implemented currently, as accessions may need to change to reference a block_group instead of a path.
Because the above mechanism doesn't actually have a path. It could be we do not have this behavior for this update style
or change accessions.

```console
gen transform --format-csv-for-gaf example.csv > example.fa 
```

This creates a fasta file suitable for alignment with graph aligners such as Minigraph, GraphAligner, VG, etc. The goal is
to obtain a GAF file.

This file will appear as such:

example.fa
```
>simple_insert_left
atgggagtataattttagatagtgaagatttctgtattcaaatgccacat
>simple_insert_right
acacagaaaaaggcaggcagagaaaataacaaggataaagacactgaagt
```

This file can then be fed to an aligner. Here are several examples:

First, we need to export to a common graph format, GFA

```console
gen export -n k562 -g k562.gfa
```

Then we can use this for various programs:

VG

Note, vg will convert node ids and the gaf must be processed so node ids are restored to their original id listed
in translations.txt
```console
vg convert -g k562.gfa -p -T translations.txt > k562.vg
vg index -g k562.gcsa -x k562.xg k562.vg
vg map --gaf -d k562 -F example.fa > example.gaf
```

GraphAligner
```console
GraphAligner -g k562.gfa -f example.fa -a aln.gaf -x vg
```

Minigraph

For minigraph, we use -w 3 to capture our shorter sequences.
```console
minigraph -c -x sr k562.gfa example.fa -w 3
```

The outputs of these programs will be a gaf file, which for the above example will be similar to this:

example.gaf
```text
simple_insert_left	50	0	49	+	>44.0	50	0	49	49	49	28	tp:A:P	NM:i:0	cm:i:15	s1:i:49	s2:i:0	dv:f:0	cg:Z:49=	ds:Z::49
simple_insert_right	50	1	50	+	>46.0	50	1	50	49	49	28	tp:A:P	NM:i:0	cm:i:14	s1:i:49	s2:i:0	dv:f:0	cg:Z:49=	ds:Z::49
```

This indicates the left arm starts at node `44.0` in our graph and aligns for 50 bases. And the right arm begins at node `46.0`. 

This GAF is then fed into gen via:

```console
gen update-gaf -n k562 --csv example.csv --gaf example.gaf --sample child
```

This will create a new sample (or reuse a sample if it exists) with the name "child". This new entry will contain a copy
of the reference graph for K562 with our sequence inserted. The insertion will be located at the end of the left homology 
arm and before the right arm. If there are misalignments in the homology arm, those regions will not be used. For 
example, suppose the CIGAR string was 45=4X, the insertion would begin after the 45th base pair. If more fine tuning is 
desired, the GAF entry can be manually modified. The location of the insertion for the left end is determined by the 
path_start field (8th column) + the number of residue matches (10th column). The right end is determined by the path start
field.
