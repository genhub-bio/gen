# Data used by `sickle_cell_hbb.ipynb`

The notebook reads three small files from this folder. Nothing is downloaded when it runs.

| File | What it holds | Source |
|------|---------------|--------|
| `hbb_reference.fa` | 6,000 bases of human chromosome 11 around the beta-globin gene | Ensembl REST API, human genome GRCh38, `11:5224001..5230000`, forward strand |
| `hbb_genes.gff3` | The *HBB* gene, its main transcript `ENST00000335295` (HBB-201), 3 exons and the coding sequence | Ensembl REST API, release 116 |
| `hbb_sickle_cell.vcf` | The sickle-cell variant rs334 in two example people | Written by hand from the Ensembl and ClinVar records below |

Accessed on 2026-11-09.

## Coordinates

The FASTA file holds one sequence named `chr11_5224001_5230000`. Base 1 of the file is chromosome 11 position 5,224,001, so a chromosome 11 position is the file position plus 5,224,000. The gene file and the variant file use the file positions.

| Item | Chromosome 11 (GRCh38) | In the files |
|------|------------------------|--------------|
| *HBB* gene, minus strand | 5,225,464 to 5,229,395 | 1,464 to 5,395 |
| HBB-201 transcript | 5,225,464 to 5,227,071 | 1,464 to 3,071 |
| First coding exon (coding part) | 5,226,930 to 5,227,021 | 2,930 to 3,021 |
| rs334, plus strand `T` to `A` | 5,227,002 | 3,002 |

The gene row ends at 5,395 because the gene record spans all of its transcripts, and the longest one (`ENST00000380315`) reaches 5,229,395. The main transcript ends at 3,071.

## The variant

rs334 is the sickle-cell variant. On the transcript it is `ENST00000335295.4:c.20A>T`, which changes codon 7 from GAG to GTG and the amino acid from glutamic acid (Glu, E) to valine (Val, V): `p.Glu7Val`. Older papers count from the amino acid after the initial methionine and call it Glu6Val. *HBB* is on the minus strand, so on the plus strand shown in the files the change is `T` to `A`.

Checked against:

- Ensembl variation record `rs334` (GRCh38 11:5227002, alleles T/A/C/G), and the Ensembl VEP consequence `missense_variant`, codons `gAg/gTg`, `p.Glu7Val`.
- ClinVar record VCV000015333, `NM_000518.5(HBB):c.20A>T (p.Glu7Val)`, GRCh38 location 11:5227002, classified Pathogenic (criteria provided, multiple submitters, no conflicts; last evaluated 2026-07-01).

The two people in the VCF file are examples made for the notebook. `sickle_cell` is homozygous for the variant and `sickle_trait` carries one copy. They are not genotypes of real individuals.

## Licence and usage terms

- Ensembl: "Ensembl imposes no restrictions on access to, or use of, the data provided" (https://www.ensembl.org/info/about/legal/disclaimer.html). EMBL-EBI adds no restrictions beyond those of the data owner and expects attribution (https://www.ebi.ac.uk/about/terms-of-use/). Please cite Ensembl, see https://www.ensembl.org/info/about/cite.html.
- ClinVar and dbSNP (NCBI): "NCBI itself places no restrictions on the use or distribution of the data contained therein." Some submitters of the original data may claim rights in it. NLM asks for acknowledgment (https://www.ncbi.nlm.nih.gov/home/about/policies/). Only the variant position, alleles and classification are used here.
- No genotypes from the 1000 Genomes Project or other population panels are used.

## Rebuilding the files

The reference and gene files come from these commands (Ensembl sometimes answers with a temporary error, so curl retries). The sequence is fixed by the genome build. The gene file reflects Ensembl release 116.

```bash
REST=https://rest.ensembl.org
CONTIG=chr11_5224001_5230000

curl -sf --retry 8 --retry-delay 5 --retry-all-errors -H 'Content-Type: text/x-fasta' \
  "$REST/sequence/region/human/11:5224001..5230000:1?coord_system_version=GRCh38" \
  | tail -n +2 | tr -d '\n' | tr a-z A-Z | fold -w 60 | { echo ">$CONTIG"; cat; echo; } | sed '/^$/d' > hbb_reference.fa

curl -sf --retry 8 --retry-delay 5 --retry-all-errors -H 'Content-Type: text/x-gff3' \
  "$REST/overlap/region/human/11:5224001..5230000?feature=gene;feature=transcript;feature=exon;feature=cds" \
  | python3 -c '
import sys
print("##gff-version 3")
exon = 0
for line in sys.stdin:
    f = line.rstrip("\n").split("\t")
    if len(f) < 9:
        continue
    kind, attributes = f[2], f[8]
    if kind == "gene" and "Name=HBB;" in attributes:
        name = "HBB"
    elif kind == "mRNA" and "ID=transcript:ENST00000335295;" in attributes:
        name = "HBB-201"
    elif "Parent=transcript:ENST00000335295;" in attributes and kind in ("exon", "CDS"):
        if kind == "exon":
            exon += 1
        name = f"HBB exon {exon}" if kind == "exon" else "HBB coding sequence"
    else:
        continue
    f[0], f[3], f[4] = "chr11_5224001_5230000", str(int(f[3]) - 5224000), str(int(f[4]) - 5224000)
    f[8] = f"Name={name}"
    print("\t".join(f))
' > hbb_genes.gff3
```
