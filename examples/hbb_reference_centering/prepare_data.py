"""Rebuild the HBB region fixtures from public sources.

Requires curl, bcftools and bgzip/tabix on PATH plus network access. The three
output files are committed alongside this script, so running it is optional.

The region is GRCh38 chr11:5,224,001-5,230,000 (1-based, inclusive). It is
written as a standalone 6,000 bp contig named HBB_region, so a position P on
chr11 becomes P - 5,224,000 in the FASTA, GFF3 and VCF.
"""

import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROM, START, END = "11", 5224001, 5230000
CONTIG = "HBB_region"
OFFSET = START - 1
SAMPLES = ["HG00096", "NA12878", "NA18501", "HG00403"]
MANE_TRANSCRIPT = "ENST00000335295"
ENSEMBL = "https://rest.ensembl.org"
VCF_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/"
    "1kGP_high_coverage_Illumina.chr11.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
)


def fetch(url, content_type):
    request = urllib.request.Request(url, headers={"Content-Type": content_type})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def build_fasta():
    url = f"{ENSEMBL}/sequence/region/human/{CHROM}:{START}..{END}:1?coord_system_version=GRCh38"
    sequence = "".join(fetch(url, "text/x-fasta").splitlines()[1:]).upper()
    assert len(sequence) == END - START + 1
    lines = [sequence[i : i + 60] for i in range(0, len(sequence), 60)]
    (HERE / "hbb_region.fa").write_text(f">{CONTIG}\n" + "\n".join(lines) + "\n")
    return sequence


def build_gff3():
    url = (
        f"{ENSEMBL}/overlap/region/human/{CHROM}:{START}..{END}"
        "?feature=gene;feature=transcript;feature=exon;feature=cds"
    )
    out = [
        "##gff-version 3",
        f"##sequence-region {CONTIG} 1 {END - OFFSET}",
        f"#!source Ensembl release REST API, GRCh38 {CHROM}:{START}-{END}",
    ]
    exon_number = 0
    for line in fetch(url, "text/x-gff3").splitlines():
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        kind, attributes = fields[2], fields[8]
        is_gene = kind == "gene" and "Name=HBB;" in attributes
        is_transcript = (
            kind == "mRNA" and f"ID=transcript:{MANE_TRANSCRIPT}" in attributes
        )
        in_transcript = f"Parent=transcript:{MANE_TRANSCRIPT}" in attributes
        if not (
            is_gene or is_transcript or (in_transcript and kind in ("exon", "CDS"))
        ):
            continue
        if kind == "gene":
            name = "HBB"
        elif kind == "mRNA":
            name = "HBB-201"
        elif kind == "exon":
            exon_number += 1
            name = f"HBB exon {exon_number}"
        else:
            name = "HBB CDS"
        fields[0] = CONTIG
        fields[3] = str(int(fields[3]) - OFFSET)
        fields[4] = str(int(fields[4]) - OFFSET)
        fields[8] = f"ID={kind}_{fields[3]};Name={name}"
        out.append("\t".join(fields))
    (HERE / "hbb_region.gff3").write_text("\n".join(out) + "\n")


def build_vcf(sequence):
    subset = subprocess.run(
        [
            "bcftools",
            "view",
            "-r",
            f"chr{CHROM}:{START}-{END}",
            "-s",
            ",".join(SAMPLES),
            "-m2",
            "-M2",
            "-v",
            "snps,indels",
            "-c",
            "1",
            VCF_URL,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    stripped = subprocess.run(
        ["bcftools", "annotate", "-x", "INFO,^FORMAT/GT"],
        input=subset,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    out = []
    for line in stripped.splitlines():
        if line.startswith("##contig") or line.startswith("##bcftools"):
            continue
        if line.startswith("#"):
            out.append(line)
            continue
        fields = line.split("\t")
        position = int(fields[1]) - OFFSET
        reference = fields[3]
        assert sequence[position - 1 : position - 1 + len(reference)] == reference, (
            fields[:5]
        )
        fields[0], fields[1], fields[2] = CONTIG, str(position), "."
        out.append("\t".join(fields))
    header_end = (
        max(index for index, line in enumerate(out) if line.startswith("##")) + 1
    )
    out.insert(header_end, f"##contig=<ID={CONTIG},length={END - OFFSET}>")
    (HERE / "hbb_variants.vcf").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    build_vcf(build_fasta())
    build_gff3()
