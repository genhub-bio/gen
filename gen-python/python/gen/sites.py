"""Breakpoints and sites, the two ways of saying where strands are cut or exchanged.

A `Breakpoint` is one cut on one strand, named by the two nucleotides it falls between. A `Site`
is a stretch of sequence that a recombinase or a restriction enzyme acts on, with the place each
strand is cut. `SequenceGraph.exchange` works with breakpoints. A site gives the breakpoints of
both its strands, so operations on sites can describe staggered cuts, and the single-strand
operations imply the antisense cut that a blunt one would have.
"""

from collections import namedtuple
from itertools import combinations
from types import MappingProxyType

from .gen import Locus, Position, SuperPosition, SuperpositionError

__all__ = ["Breakpoint", "Motif", "Site", "recombination_closure", "site_pairs"]


def _one_position(value, role):
    if isinstance(value, SuperPosition):
        raise SuperpositionError(
            f"a breakpoint needs one Position for its {role} side, but got a SuperPosition. "
            "A step next to a fork gives one: name the position on the branch you mean"
        )
    if not isinstance(value, Position):
        raise TypeError(f"a breakpoint needs a Position for its {role} side, not {type(value).__name__}")
    return value


_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def _reverse_complement(sequence):
    return sequence.translate(_COMPLEMENT)[::-1]


def _matches(pattern, sequence):
    """Whether `sequence` is an instance of the IUPAC `pattern`, base for base."""
    return len(pattern) == len(sequence) and all(
        base in _IUPAC[code] for code, base in zip(pattern, sequence.upper())
    )


class Breakpoint(namedtuple("Breakpoint", ["upstream", "downstream"])):
    """A cut between two nucleotides, read upstream to downstream on one strand.

    A breakpoint is a pair, so it can be unpacked and passed anywhere a pair of positions can. It
    does not say whether the graph joins the two: that is checked by the operation that uses it.
    Inside a node on one strand the graph always does, and elsewhere it takes an edge, which may
    cross strands. A `SuperPosition` on either side raises `SuperpositionError`.
    """

    __slots__ = ()

    def __new__(cls, upstream, downstream):
        return super().__new__(
            cls, _one_position(upstream, "upstream"), _one_position(downstream, "downstream")
        )

    @classmethod
    def after(cls, position):
        """The cut just after `position`, between it and the next nucleotide on its strand."""
        return cls(position, position + 1)

    @property
    def strand(self):
        """The strand the cut is read on, which is that of its upstream nucleotide."""
        return self.upstream.strand

    def reverse_complement(self):
        """The same cut on the opposite strand: the two nucleotides swap places and flip."""
        return Breakpoint(self.downstream.reverse_complement(), self.upstream.reverse_complement())

    def on(self, graph):
        """The same cut with both nucleotides attached to `graph`."""
        return Breakpoint(self.upstream.on(graph), self.downstream.on(graph))


class Motif:
    """The definition of a site: what an enzyme recognises and where it cuts inside it.

    `recognition_site` is the IUPAC pattern to search for, so `AANNTT` stands for every six-base
    site that begins `AA` and ends `TT`. `top` and `bottom` count the cuts along that pattern as
    they do for a `Site`, and `bottom` defaults to `top` for a blunt cut. A motif holds no graph.
    Searching with `recognition_site` finds loci, and `apply` turns each locus into the `Site` that
    this motif defines there, which is where the concrete bases and graph positions come in::

        hits = sequence_graph.search(lox.recognition_site)
        sites = [lox.apply(hit) for hit in hits]

    `parts` optionally names stretches of the recognition site as `(start, end)` pairs, counted
    from the start of the pattern like the cuts. A recombinase with arms around a spacer can name
    them, such as `{"left": (0, 13), "spacer": (13, 21), "right": (21, 34)}`, and `Site.part`
    then gives each stretch of a concrete site, so two sites can be compared part by part as they
    are found. Nothing requires parts, and a motif with other structure, or none, names its own.
    """

    __slots__ = ("recognition_site", "top", "bottom", "name", "parts")

    def __init__(self, recognition_site, top, bottom=None, name=None, parts=None):
        pattern = recognition_site.upper()
        unknown = sorted(set(pattern) - set(_IUPAC))
        if unknown:
            raise ValueError(f"a recognition site uses IUPAC codes, not {''.join(unknown)!r}")
        bottom = top if bottom is None else bottom
        for label, cut in (("top", top), ("bottom", bottom)):
            if not isinstance(cut, int) or isinstance(cut, bool):
                raise TypeError(f"the {label} cut must be an integer, not {type(cut).__name__}")
            if not 0 <= cut <= len(pattern):
                raise ValueError(
                    f"the {label} cut {cut} lies outside a recognition site of {len(pattern)} bases"
                )
        parts = {} if parts is None else dict(parts)
        for part, span in parts.items():
            start, end = span
            if not (isinstance(start, int) and isinstance(end, int)) or not 0 <= start <= end <= len(pattern):
                raise ValueError(
                    f"the part {part!r} spans {span!r}, which is not a start and end inside a "
                    f"recognition site of {len(pattern)} bases"
                )
            parts[part] = (start, end)
        self.recognition_site = pattern
        self.top = top
        self.bottom = bottom
        self.name = name
        self.parts = MappingProxyType(parts)

    @property
    def palindromic(self):
        """Whether the recognition site reads the same on both strands, so a hit has no direction."""
        return _reverse_complement(self.recognition_site) == self.recognition_site

    def apply(self, locus):
        """The `Site` this motif defines on `locus`, read in the motif's own direction.

        A locus that spells the recognition site is used as it is. One that spells its reverse
        complement is reversed first, so the cuts always count along the motif. Raises `ValueError`
        when the locus is neither.
        """
        sequence = locus.sequence
        if _matches(self.recognition_site, sequence):
            return Site(locus, self.top, self.bottom, motif=self)
        if _matches(self.recognition_site, _reverse_complement(sequence)):
            return Site(locus.reverse_complement(), self.top, self.bottom, motif=self)
        raise ValueError(
            f"the locus {sequence!r} is not an instance of the recognition site {self.recognition_site!r}"
        )

    def __repr__(self):
        label = f"{self.name!r}, " if self.name else ""
        return f"Motif({label}{self.recognition_site!r}, top={self.top}, bottom={self.bottom})"


class Site:
    """A stretch of sequence an enzyme acts on, and where each strand is cut inside it.

    `locus` is the oriented string the site was found as. `top` is the cut on the strand the locus
    reads on and `bottom` the cut on the other strand, both counted in bases from the start of the
    locus in its reading direction, so a cut of 3 falls between the third and fourth base. Equal
    cuts are blunt. When the bottom cut lies further along, each end keeps a 5' overhang of the
    difference, and a negative difference leaves 3' overhangs. Left out, `bottom` is `top`.

    The cuts are resolved to `Breakpoint`s of graph positions when the site is made (`top_break` and
    `bottom_break`), so a cut at the edge of the locus that sits next to a fork, where the
    neighbouring base is ambiguous, raises `SuperpositionError` at once instead of when the site
    is used.

    Counting along the locus keeps the site independent of the strands of its individual bases, so
    a site that crosses an inversion works like any other.
    """

    __slots__ = ("locus", "top", "bottom", "motif", "top_break", "bottom_break")

    def __init__(self, locus, top, bottom=None, motif=None):
        if not isinstance(locus, Locus):
            raise TypeError(f"a site needs a Locus, not {type(locus).__name__}")
        bottom = top if bottom is None else bottom
        for name, cut in (("top", top), ("bottom", bottom)):
            if not isinstance(cut, int) or isinstance(cut, bool):
                raise TypeError(f"the {name} cut must be an integer, not {type(cut).__name__}")
            if not 0 <= cut <= len(locus):
                raise ValueError(
                    f"the {name} cut {cut} lies outside a site of {len(locus)} bases; "
                    f"a cut counts bases from the start, from 0 to {len(locus)}"
                )
        self.locus = locus
        self.top = top
        self.bottom = bottom
        self.motif = motif
        self.top_break, self.bottom_break = self._breakpoints()

    @property
    def blunt(self):
        return self.top == self.bottom

    @property
    def overhang(self):
        """How far the bottom cut lies beyond the top one: positive is a 5' overhang."""
        return self.bottom - self.top

    def _before(self, cut):
        return self.locus[cut - 1] if cut > 0 else self.locus[0] - 1

    def _after(self, cut):
        return self.locus[cut] if cut < len(self.locus) else self.locus[len(self.locus) - 1] + 1

    def _breakpoints(self):
        top = Breakpoint(self._before(self.top), self._after(self.top))
        bottom = Breakpoint(
            self._after(self.bottom).reverse_complement(),
            self._before(self.bottom).reverse_complement(),
        )
        return top, bottom

    def breakpoints(self):
        """The top-strand and bottom-strand breakpoints, as a pair of `Breakpoint`s.

        They are fixed when the site is made, so a cut that has no single position on either side
        raises `SuperpositionError` there, and later edits to the graph do not move them. The bottom
        one is read on the other strand, from the nucleotide after the cut to the one before it, so
        a blunt site has the two cuts as mirror images.
        """
        return self.top_break, self.bottom_break

    def reverse_complement(self):
        """The same site read the other way: the locus reverses and the cuts swap strands.

        The cuts stay on the same bases, counted from the other end. A site that is not its own
        reverse complement is the same site either way. A palindromic one, such as a LoxPsym site
        that reads the same on both strands, has no direction of its own, so this is how a caller
        says which way round two such sites are aligned.
        """
        length = len(self.locus)
        return Site(
            self.locus.reverse_complement(), length - self.bottom, length - self.top, self.motif
        )

    @property
    def palindromic(self):
        """Whether the site's bases read the same on both strands, so it has no direction of its own."""
        return _reverse_complement(self.locus.sequence) == self.locus.sequence

    def part(self, name):
        """The stretch of this site that its motif names `name`, read in the locus's direction.

        A site made from a motif with `parts` can be read part by part, for example to compare the
        spacers of two sites. The stretch follows the site's own reading direction, so on a site
        that reads as the reverse complement of the motif, a part counted from the start of the
        recognition site is counted from the end of the locus instead. Raises `KeyError` for a
        name the motif does not give.
        """
        if self.motif is None or name not in self.motif.parts:
            raise KeyError(f"the site's motif has no part named {name!r}")
        start, end = self.motif.parts[name]
        length = len(self.locus)
        if not _matches(self.motif.recognition_site, self.locus.sequence):
            start, end = length - end, length - start
        return self.locus.slice(start, end)

    def _stretch(self):
        """The stretch of graph positions the site covers, the same from either strand."""
        return frozenset((self.locus, self.locus.reverse_complement()))

    def compatible(self, other):
        """Whether exchanging this site with `other` leaves ends that anneal."""
        try:
            self.check_overhang_matches(other)
        except ValueError:
            return False
        return True

    def overhang_sequence(self):
        """The bases the two cuts leave single-stranded, in the reading direction of the locus."""
        low, high = sorted((self.top, self.bottom))
        return self.locus.slice(low, high).sequence if high > low else ""

    def check_overhang_matches(self, other):
        """Raise `ValueError` unless exchanging this site with `other` leaves ends that anneal.

        Blunt sites always do. Staggered ones need the same overhang, base for base, because each
        product carries one strand's overhang from each parent, and different bases would leave a
        bubble in the double strand.
        """
        if self.overhang != other.overhang:
            raise ValueError(
                f"the sites cut with overhangs of {self.overhang} and {other.overhang} bases, "
                "so the exchanged ends would not anneal"
            )
        ours, theirs = self.overhang_sequence(), other.overhang_sequence()
        if ours != theirs:
            raise ValueError(
                f"the overhangs {ours} and {theirs} differ, so the exchanged ends would not "
                "anneal and the products would carry a bubble"
            )

    def __eq__(self, other):
        return (
            isinstance(other, Site)
            and (self.locus, self.top, self.bottom) == (other.locus, other.top, other.bottom)
        )

    def __hash__(self):
        return hash((self.locus, self.top, self.bottom))

    def __repr__(self):
        return f"Site({self.locus!r}, top={self.top}, bottom={self.bottom})"


def site_pairs(sites):
    """Every compatible pair of distinct physical sites, with each alignment a palindrome allows.

    A search for a palindromic recognition site returns each hit once per reading, so the two
    readings are one site here and never pair with each other. A pair of palindromic sites is
    offered in both alignments, since one excises and the other inverts. A site with a direction
    has the alignment its bases fix.
    """
    unique = {}
    for site in sites:
        unique.setdefault(site._stretch(), site)
    for first, second in combinations(unique.values(), 2):
        if not first.compatible(second):
            continue
        yield first, second
        if second.palindromic:
            yield first, second.reverse_complement()


def recombination_closure(sequence_graph, motif, *, stack=True, compatible=None):
    """Recombine every pair of sites of `motif`, searching again until no new pair appears.

    A reaction can write a junction that is itself a site, so the search repeats. A pair is tried
    once, whichever alignment it took. Pairs whose cuts a stacked graph does not join, because they
    lie on alternatives that never share a molecule, are skipped. `compatible(first, second)`
    narrows the pairs further, for rules such as attP with attB only. Returns the number of pairs
    tried and the number skipped.
    """
    tried, skipped = set(), 0
    while True:
        sites = _find_sites(sequence_graph, motif)
        pending = {}
        for first, second in site_pairs(sites):
            pair = frozenset((first._stretch(), second._stretch()))
            if pair in tried or (compatible is not None and not compatible(first, second)):
                continue
            pending.setdefault(pair, []).append((first, second))
        if not pending:
            return len(tried), skipped
        for pair, alignments in pending.items():
            tried.add(pair)
            for first, second in alignments:
                try:
                    sequence_graph.recombine(first, second, stack=stack)
                except ValueError as error:
                    if "not joined" not in str(error):
                        raise
                    skipped += 1


def _find_sites(sequence_graph, motif):
    """The sites of `motif` in the graph, each read in the motif's direction.

    A double-stranded search reports a hit on the reverse strand as a locus that can spell the
    wrong bases once it crosses an inversion junction, and `apply` rejects it. Until that is fixed,
    a recognition site of plain ACGT is searched as itself and as its reverse complement, each on
    the forward strand, and the second set of hits is reversed to read along the motif.
    """
    pattern = motif.recognition_site
    if set(pattern) - set("ACGT"):
        return [motif.apply(hit) for hit in sequence_graph.search(pattern)]
    hits = sequence_graph.search(pattern, sequence_kind="exact")
    if not motif.palindromic:
        opposite = sequence_graph.search(_reverse_complement(pattern), sequence_kind="exact")
        hits += [hit.reverse_complement() for hit in opposite]
    return [motif.apply(hit) for hit in hits]
