# Tracking sequence edits across applications - without plugins

In this example a user makes changes in a third party genetic sequence editing suite like the freely available [ApE](https://jorgensen.biology.utah.edu/wayned/ape/) or commercial software like [Geneious](https://www.geneious.com/). In order to track these changes in Gen as well, we need a way to represent what the user did in a format gen can understand. What follows is one possible route, but this is still very much a work in progress.

The task at hand is: "change the selection marker of the Bacterial Artificial Chromosome (BAC) pBeloBAC11 ([file](./pBeloBAC11.gb) provided by NEB) from chloramphenicol resistance to kanamycin resistance." We will use the kanamycin resistance marker from the Tn5 transposon (GenBank: U00004.1), which we can download from NCBI:

```console
wget "https://github.com/genhub-bio/gen/blob/7814933b99515fc2c7749b7df3014cf438a4ddf9/examples/externally_edited_files/pBeloBAC11.gb?raw=true" -O pBeloBAC11.gb
wget "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&rettype=gb&id=U00004.1" -O Tn5.gb
```

We then initialize a new gen repository and import the BAC sequence directly from GenBank.
```console
gen init
gen import --gb pBeloBAC11.gb
```

Save a copy of the original sequence before editing — we'll need it as the reference in the pairwise alignment:
```console
mkdir snapshot_1
cp pBeloBAC11.* snapshot_1
```

Then we move over to our sequence editor, where we edit the BAC sequence to replace the chloramphenicol acetyl transferase (cat) coding sequence by the coding sequence of aminoglycoside-3'-O-phosphotransferase from Tn5 (don't forget to reverse complement it before pasting). We then save the resulting sequence, overwriting the original file.

We now have two files called pBeloBAC11.gb, one in the working directory, and one in the snapshot_1 directory. To represent the changes in a format wfmash and seqwish can process, convert both to FASTA:
```console
python -c "from Bio import SeqIO; SeqIO.convert('snapshot_1/pBeloBAC11.gb', 'genbank', 'snapshot_1/pBeloBAC11.fa', 'fasta')"
python -c "from Bio import SeqIO; SeqIO.convert('pBeloBAC11.gb', 'genbank', 'pBeloBAC11.fa', 'fasta')"
```

Next, we perform a pairwise alignment using [wfmash](https://github.com/waveygang/wfmash), and graph induction using [seqwish](https://github.com/ekg/seqwish). Both applications are conveniently available in the [PanGenome Graph Builder (pggb)](https://github.com/pangenome/pggb) Docker image that can be launched as follows:

```console
docker run -it -v $PWD:/data -w /data ghcr.io/pangenome/pggb:latest
```

First we concatenate the old and new sequence, renaming them to _ref_ and _alt_, respectively. The resulting file is indexed using samtools and used as input to wfmash. The obtained alignment (aln.paf) is then passed to seqwish to build the graph. By specifying a --min-match-len of 12, we control the complexity of the graph topology induced from the alignment.

```console
(sed 's/^>.*/>ref/' snapshot_1/pBeloBAC11.fa; sed 's/^>.*/>alt/' pBeloBAC11.fa) > combined.fa
samtools faidx combined.fa
wfmash combined.fa > aln.paf
seqwish --seqs combined.fa --paf-alns aln.paf --gfa out.gfa --min-match-len=12
```

In this specific example ommitting the min-match-len filter results in a graph that contains a node of only 1 bp long because of a local alignment. This is confirmed by the CIGAR string in the alignment file: `cg:Z:766=790I1=655D6085=`. If further normalization is needed the complete pggb pipeline can be used instead, but it was not necessary in this case.

The path lines in the GFA file indicate that this is a replacement: node 2 is replaced by node 4. Gen doesn't know about these node identifiers however, so we can't import the GFA directly at this point. One option is to convert the path differences to one or more gen update commands, an alternative approach may be to rewrite the node IDs. (TODO) 
```
P       ref     1+,2+,3+        *
P       alt     1+,4+,3+        *
```

## Appendix
### Minimap2 instead of wfmash
You can also use minimap2 as an aligner instead of wfmash. The Docker image referenced above does not include the minimap2 binary, but it can be downloaded as follows:

```console
curl -L https://github.com/lh3/minimap2/releases/download/v2.28/minimap2-2.28_x64-linux.tar.bz2 | tar -jxvf - minimap2-2.28_x64-linux/minimap2 --strip-components 1
```

Minimap2 doesn't use a combined fasta file like wfmash, so we make two files for the different versions of pBeloBAC11. Seqwish runs the same as above, and the resulting GFA file is identical.

```console
sed 's/^>.*/>ref/' ./snapshot_1/pBeloBAC11.fa > ref.fa
sed 's/^>.*/>alt/' pBeloBAC11.fa > alt.fa
./minimap2 -x asm5 -c -o aln.paf alt.fa ref.fa
cat ref.fa alt.fa > combined.fa
seqwish --seqs combined.fa --paf-alns aln.paf --gfa out.gfa --min-match-len=12
```
