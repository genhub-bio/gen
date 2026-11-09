"""Download the SARS-CoV-2 reference, its annotations, and build a lineage VCF.

Sources:
  * NC_045512.2 FASTA and GFF3 from NCBI Entrez efetch.
  * Lineage-defining mutations from the cov-lineages constellations project
    (https://github.com/cov-lineages/constellations), which lists amino acid
    and nucleotide changes per lineage. Amino acid substitutions are converted
    to the single nucleotide change that produces them in the reference codon.

Usage: python build_data.py
"""

import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACCESSION = "NC_045512.2"
EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id={acc}&rettype={rettype}&retmode=text"
)
CONSTELLATION = (
    "https://raw.githubusercontent.com/cov-lineages/constellations/main/"
    "constellations/definitions/c{lineage}.json"
)
# Alpha deletions given only at amino acid level upstream: ORF1ab SGF3675-3677
# (nt 11288-11296), spike H69-V70 (nt 21765-21770), and spike Y144 (nt 21991-21993).
EXTRA_SITES = {"B.1.1.7": ["del:11288:9", "del:21765:6", "del:21991:3"]}
SAMPLES = {"Alpha": "B.1.1.7", "Delta": "B.1.617.2", "Omicron_BA.1": "BA.1"}

# CDS start (1-based) for the genes referenced by the definitions.
GENE_STARTS = {
    "1ab": 266,
    "orf1ab": 266,
    "s": 21563,
    "spike": 21563,
    "orf3a": 25393,
    "3a": 25393,
    "e": 26245,
    "m": 26523,
    "orf6": 27202,
    "6": 27202,
    "orf7a": 27394,
    "7a": 27394,
    "orf7b": 27756,
    "orf8": 27894,
    "8": 27894,
    "n": 28274,
}
CODON_TABLE = dict(
    zip(
        (a + b + c for a in "TCAG" for b in "TCAG" for c in "TCAG"),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    )
)


def fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode()


def read_fasta(path):
    lines = path.read_text().splitlines()
    return "".join(lines[1:]).upper()


def substitution_to_snp(reference, gene, ref_aa, codon, alt_aa):
    """Return (pos, ref, alt) for the unique single-nucleotide change, else None."""
    start = GENE_STARTS[gene.lower()]
    if gene.lower() in ("1ab", "orf1ab") and codon > 4401:
        return None  # beyond the frameshift; skip
    pos = start + (codon - 1) * 3
    ref_codon = reference[pos - 1 : pos + 2]
    if CODON_TABLE[ref_codon] != ref_aa:
        raise ValueError(f"{gene}:{ref_aa}{codon} does not match reference {ref_codon}")
    candidates = []
    for offset in range(3):
        for base in "ACGT":
            if base == ref_codon[offset]:
                continue
            mutated = ref_codon[:offset] + base + ref_codon[offset + 1 :]
            if CODON_TABLE[mutated] == alt_aa:
                candidates.append((pos + offset, ref_codon[offset], base))
    return candidates[0] if len(candidates) == 1 else None


def lineage_variants(reference, sites):
    """Return {(pos, ref): alt} for a lineage's definition, skipping ambiguous entries."""
    variants = {}
    skipped = []
    for site in sites:
        nuc = re.fullmatch(r"nuc:([ACGT])(\d+)([ACGT])", site)
        deletion = re.fullmatch(r"del:(\d+):(\d+)", site)
        insertion = re.fullmatch(r"nuc:(\d+)\+([ACGT]+)", site)
        substitution = re.fullmatch(r"([^:]+):([A-Z])(\d+)([A-Z])", site)
        if nuc:
            variants[(int(nuc[2]), nuc[1])] = nuc[3]
        elif deletion:
            pos, length = int(deletion[1]), int(deletion[2])
            anchor = pos - 1
            variants[(anchor, reference[anchor - 1 : anchor + length])] = reference[
                anchor - 1
            ]
        elif insertion:
            anchor = int(insertion[1])
            base = reference[anchor - 1]
            variants[(anchor, base)] = base + insertion[2]
        elif substitution:
            snp = substitution_to_snp(
                reference,
                substitution[1],
                substitution[2],
                int(substitution[3]),
                substitution[4],
            )
            if snp:
                variants[(snp[0], snp[1])] = snp[2]
            else:
                skipped.append(site)
        else:
            skipped.append(site)
    return variants, skipped


def main():
    fasta_path = HERE / f"{ACCESSION}.fa"
    fasta_path.write_text(fetch(EFETCH.format(acc=ACCESSION, rettype="fasta")))
    (HERE / f"{ACCESSION}.gff3").write_text(
        fetch(EFETCH.format(acc=ACCESSION, rettype="gff3"))
    )
    reference = read_fasta(fasta_path)

    per_sample = {}
    for sample, lineage in SAMPLES.items():
        sites = json.loads(fetch(CONSTELLATION.format(lineage=lineage)))["sites"]
        sites += EXTRA_SITES.get(lineage, [])
        per_sample[sample], skipped = lineage_variants(reference, sites)
        print(f"{sample}: {len(per_sample[sample])} variants; skipped {skipped}")

    names = list(SAMPLES)
    records = {}
    for sample, variants in per_sample.items():
        for (pos, ref), alt in variants.items():
            assert reference[pos - 1 : pos - 1 + len(ref)] == ref, (sample, pos, ref)
            records.setdefault((pos, ref), {})[sample] = alt

    lines = [
        "##fileformat=VCFv4.2",
        f"##contig=<ID={ACCESSION},length={len(reference)}>",
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(names),
    ]
    for (pos, ref), by_sample in sorted(records.items()):
        alts = sorted(set(by_sample.values()))
        genotypes = [
            f"{alts.index(by_sample[name]) + 1}/{alts.index(by_sample[name]) + 1}"
            if name in by_sample
            else "0/0"
            for name in names
        ]
        lines.append(
            f"{ACCESSION}\t{pos}\t.\t{ref}\t{','.join(alts)}\t.\tPASS\t.\tGT\t"
            + "\t".join(genotypes)
        )
    (HERE / "lineages.vcf").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(records)} records")


if __name__ == "__main__":
    main()
