"""Task definitions for the CLI-first vs Python-first skill eval.

Each task copies fixture files into an isolated working directory, gives the
agent a plain-English intent, and verifies the resulting `.gen` repository
against ground truth computed independently via the `gen` Python API (not by
trusting whatever the agent produced).
"""

import dataclasses
import pathlib
import shutil

import gen

FIXTURES_DIR = pathlib.Path(__file__).resolve().parents[1] / "fixtures"


@dataclasses.dataclass
class Task:
    name: str
    intent: str
    fixtures: list[str]

    # verify(workdir) -> (passed, message). Opens the repo the agent produced
    # at `workdir` (agents are told to use the working directory itself as
    # the repository root) and asserts on it.
    verify: object

    def setup(self, workdir: pathlib.Path) -> None:
        for fixture in self.fixtures:
            shutil.copy(FIXTURES_DIR / fixture, workdir / fixture)


def _open(workdir: pathlib.Path) -> gen.Repository:
    return gen.Repository(str(workdir))


def _exported_sequence(
    repo: gen.Repository, workdir: pathlib.Path, sample_name: str
) -> str:
    out = workdir / f"_verify_{sample_name}.fa"
    repo.export_fasta(str(out), sample=sample_name)
    return "".join(
        line.strip()
        for line in out.read_text().splitlines()
        if not line.startswith(">")
    )


def _verify_import_fasta(workdir: pathlib.Path):
    repo = _open(workdir)
    samples = {s.sample_name: s for s in repo.get_samples()}
    if "ref" not in samples:
        return False, f"no sample named 'ref' found; samples present: {list(samples)}"
    sample = samples["ref"]
    if len(sample) != 1 or sample[0].name != "m123":
        return (
            False,
            f"expected one graph named 'm123' in sample 'ref', got {[g.name for g in sample]}",
        )
    sequence = _exported_sequence(repo, workdir, "ref")
    expected = "ATCGATCGATCGATCGATCGGGAACACACAGAGA"
    if sequence != expected:
        return False, f"exported sequence {sequence!r} != expected {expected!r}"
    return True, "ok"


def _verify_region_replace(workdir: pathlib.Path):
    repo = _open(workdir)
    samples = {s.sample_name: s for s in repo.get_samples()}
    if "ref" not in samples or "edited" not in samples:
        return False, f"expected samples 'ref' and 'edited'; got {list(samples)}"

    ref_sequence = _exported_sequence(repo, workdir, "ref")
    edited_sequence = _exported_sequence(repo, workdir, "edited")
    if ref_sequence != "ATCGATCGATCGATCGATCGGGAACACACAGAGA":
        return False, f"'ref' sequence changed unexpectedly: {ref_sequence!r}"
    if edited_sequence != "ATCTTTTTTTTCGATCGATCGGGAACACACAGAGA":
        return (
            False,
            f"'edited' sequence {edited_sequence!r} != expected 'ATCTTTTTTTTCGATCGATCGGGAACACACAGAGA'",
        )
    return True, "ok"


def _verify_vcf_update(workdir: pathlib.Path):
    repo = _open(workdir)
    samples = {s.sample_name: s for s in repo.get_samples()}
    if "ref" not in samples:
        return False, f"no sample named 'ref' found; samples present: {list(samples)}"
    derived = [name for name in samples if name != "ref"]
    if not derived:
        return False, "no sample derived from the VCF update was found"
    for name in derived:
        graph = samples[name][0]
        hits = repo.search("TAGA", bgs=[graph], sequence_kind="exact")
        if hits and hits[0][1]:
            return True, f"found the VCF insertion allele in derived sample {name!r}"
    return (
        False,
        f"none of the derived samples {derived} contain the VCF insertion allele (TAGA)",
    )


def _verify_combinatorial_library(workdir: pathlib.Path):
    repo = _open(workdir)
    graphs = repo.get_sequence_graphs()
    if not graphs:
        return False, "no sequence graphs found"
    part_sequences = ["AAAA", "TAAT", "CAAC", "ATGATAA", "ATGTTAA", "ATGCTAA"]
    for graph in graphs:
        missing = [
            seq
            for seq in part_sequences
            if not repo.search(seq, bgs=[graph], sequence_kind="exact")
        ]
        if missing:
            continue
        try:
            nx_graph = graph.to_networkx()
            sources = [n for n, d in nx_graph.in_degree() if d == 0]
            sinks = [n for n, d in nx_graph.out_degree() if d == 0]
            import networkx as nx

            path_count = sum(
                1
                for source in sources
                for sink in sinks
                for _ in nx.all_simple_paths(nx_graph, source, sink)
            )
        except ImportError:
            return (
                True,
                "all 6 part sequences found (networkx unavailable to confirm 9 combinations)",
            )
        if path_count == 9:
            return (
                True,
                "all 6 part sequences found and graph has 9 combinatorial paths",
            )
        return (
            False,
            f"found all part sequences but path count was {path_count}, expected 9",
        )
    return (
        False,
        "no graph contains all 6 expected part sequences (AAAA/TAAT/CAAC/ATGATAA/ATGTTAA/ATGCTAA)",
    )


# The pUC19 bla / AmpR gene translated with the standard genetic code, derived
# independently from the fixture's CDS (not via gen). The agent is free to
# translate however it likes, so we grade the biological outcome — the exact
# beta-lactamase protein — rather than gen's translation mechanism.
AMPR_PROTEIN = "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRIDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPVAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"


def _graph_sequence(graph) -> str:
    # translate_annotation output has no current path, so export_fasta can't
    # read it — walk the graph in topological order instead. Raises on a cyclic
    # graph (e.g. a circular DNA plasmid), which the caller skips.
    import networkx

    ordered = networkx.topological_sort(graph.to_networkx())
    return "".join(graph.get_node_sequence(node) for node in ordered)


def _verify_translate_annotation(workdir: pathlib.Path):
    repo = _open(workdir)
    for sample in repo.get_samples():
        for graph in sample:
            try:
                sequence = _graph_sequence(graph)
            except Exception:
                continue
            if AMPR_PROTEIN in sequence.replace("*", ""):
                return (
                    True,
                    f"sample {sample.sample_name!r} graph {graph.name!r} holds the AmpR/bla protein translation",
                )
    return (
        False,
        "no sample holds the AmpR/bla protein translation (expected the beta-lactamase amino-acid sequence)",
    )


TASKS: list[Task] = [
    Task(
        name="import-fasta",
        intent=(
            "In this working directory, set up a gen repository and import simple.fa "
            "as a sample named 'ref'."
        ),
        fixtures=["simple.fa"],
        verify=_verify_import_fasta,
    ),
    Task(
        name="region-replace",
        intent=(
            "In this working directory, set up a gen repository and import simple.fa as "
            "sample 'ref'. Then create a new sample named 'edited', derived from 'ref', "
            "that replaces the region 3-10 (0-based, half-open) of the m123 sequence with "
            "the literal sequence TTTTTTTT. 'ref' itself must remain unmodified."
        ),
        fixtures=["simple.fa"],
        verify=_verify_region_replace,
    ),
    Task(
        name="vcf-update",
        intent=(
            "In this working directory, set up a gen repository and import simple.fa as "
            "sample 'ref'. Then apply the variants in simple.vcf against 'ref' to create "
            "the branch(es) described by the VCF's sample genotype columns."
        ),
        fixtures=["simple.fa", "simple.vcf"],
        verify=_verify_vcf_update,
    ),
    Task(
        name="combinatorial-library",
        intent=(
            "In this working directory, set up a gen repository. Build a combinatorial "
            "library named 'my_library' from the parts in parts.fa using the column "
            "layout described in combinatorial_design.csv."
        ),
        fixtures=["parts.fa", "combinatorial_design.csv"],
        verify=_verify_combinatorial_library,
    ),
    Task(
        name="translate-annotation",
        intent=(
            "In this working directory, set up a gen repository and import puc19.gb as "
            "sample 'wt'. Find the annotation for the AmpR / bla gene and translate it "
            "into a new protein sequence graph."
        ),
        fixtures=["puc19.gb"],
        verify=_verify_translate_annotation,
    ),
]
