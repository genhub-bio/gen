"""Breakpoints and sites: where strands are cut or exchanged."""

from test_editing import EditingTestCase, reverse_complement

import gen
from networkx import is_directed_acyclic_graph


class SitesTestCase(EditingTestCase):
    def sequences(self, graph):
        return sorted(str(sequence) for sequence in graph.all_sequences())

    def line(self):
        graph = self.repository.import_sequence("AAACCGTTT", name="line", sample="sites")

        def base(index):
            return graph.locus(index, index + 1).start()

        return graph, base


class BreakpointTests(SitesTestCase):
    def test_a_breakpoint_after_a_position_is_that_position_and_the_next(self):
        graph, base = self.line()

        breakpoint = gen.Breakpoint.after(base(2))

        self.assertEqual(tuple(breakpoint), (base(2), base(3)))
        self.assertEqual(breakpoint.strand, "+")

    def test_a_breakpoint_next_to_a_fork_raises_a_superposition_error(self):
        graph, base = self.line()
        graph.exchange((base(2), base(3)), (base(4), base(5)), stack=True)

        with self.assertRaises(gen.SuperpositionError):
            gen.Breakpoint.after(base(2))
        with self.assertRaises(gen.SuperpositionError):
            gen.Breakpoint(base(2), base(2) + 1)

    def test_reversing_a_breakpoint_swaps_and_flips_its_nucleotides(self):
        graph, base = self.line()

        reverse = gen.Breakpoint(base(2), base(3)).reverse_complement()

        self.assertEqual(
            tuple(reverse),
            (base(3).reverse_complement(), base(2).reverse_complement()),
        )
        self.assertEqual(reverse.strand, "-")

    def test_a_breakpoint_is_accepted_wherever_a_pair_is(self):
        graph, base = self.line()

        graph.exchange(
            gen.Breakpoint(base(2), base(3)),
            gen.Breakpoint(base(5), base(6)).reverse_complement(),
        )

        self.assertEqual(self.sequences(graph), ["AAACGGTTT"])

    def test_a_breakpoint_needs_positions(self):
        with self.assertRaises(TypeError):
            gen.Breakpoint(1, 2)


class SiteTests(SitesTestCase):
    def test_a_blunt_site_has_mirrored_cuts(self):
        graph, base = self.line()

        site = gen.Site(graph.locus(2, 7), 3)
        top, bottom = site.breakpoints()

        self.assertTrue(site.blunt)
        self.assertEqual(site.overhang, 0)
        self.assertEqual(tuple(top), (base(4), base(5)))
        self.assertEqual(bottom, top.reverse_complement())

    def test_a_staggered_site_cuts_each_strand_where_it_says(self):
        graph, base = self.line()

        site = gen.Site(graph.locus(2, 7), 1, 3)
        top, bottom = site.breakpoints()

        self.assertFalse(site.blunt)
        self.assertEqual(site.overhang, 2)
        self.assertEqual(tuple(top), (base(2), base(3)))
        self.assertEqual(
            tuple(bottom), (base(5).reverse_complement(), base(4).reverse_complement())
        )

    def test_a_cut_at_the_edge_of_a_site_falls_beside_it(self):
        graph, base = self.line()

        top, _ = gen.Site(graph.locus(2, 7), 0).breakpoints()

        self.assertEqual(tuple(top), (base(1), base(2)))

    def test_a_cut_outside_the_site_is_rejected(self):
        graph, base = self.line()

        with self.assertRaisesRegex(ValueError, "outside a site"):
            gen.Site(graph.locus(2, 7), 6)

    def test_a_site_needs_a_locus(self):
        with self.assertRaises(TypeError):
            gen.Site("ACGT", 2)

    def test_a_site_counts_cuts_along_a_hit_that_crosses_an_inversion(self):
        graph, base = self.line()
        graph.exchange(
            (base(2), base(3)),
            (base(6).reverse_complement(), base(5).reverse_complement()),
        )
        (hit,) = graph.search("AACGG", sequence_kind="exact")
        self.assertEqual(hit.strand, "mixed")

        site = gen.Site(hit, 3)
        top, bottom = site.breakpoints()

        self.assertEqual(hit.direction, "+")
        self.assertEqual(bottom, top.reverse_complement())
        self.assertEqual(top.upstream, hit[2])
        self.assertEqual(top.downstream, hit[3])


class LocusDirectionTests(SitesTestCase):
    def test_a_locus_read_as_given_runs_forward_and_its_reverse_complement_backward(self):
        graph, base = self.line()
        locus = graph.locus(0, 3)

        self.assertEqual(locus.direction, "+")
        self.assertEqual(locus.reverse_complement().direction, "-")
        self.assertEqual(locus.reverse_complement().reverse_complement().direction, "+")

    def test_a_slice_keeps_the_direction_of_its_locus(self):
        graph, base = self.line()

        self.assertEqual(graph.locus(0, 5).reverse_complement()[1:3].direction, "-")

    def test_a_search_hit_on_the_reverse_complement_runs_backward(self):
        graph, base = self.line()

        directions = {hit.direction for hit in graph.search("GTTT")}

        self.assertEqual(directions, {"+", "-"})


class RecombineTests(SitesTestCase):
    LEFT = "ATAACTTCGTATAATGT"
    RIGHT = "ATGCTATACGAAGTTAT"
    SITE = LEFT + RIGHT

    def site_in(self, graph, start, length=None, top=17, bottom=None):
        length = length or len(self.SITE)
        return gen.Site(graph.locus(start, start + length), top, bottom)

    def test_two_linear_molecules_cross_their_arms_over(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        b = self.repository.import_sequence("AAAA" + self.SITE + "TTTT", name="B", sample="s")

        products = a.recombine(self.site_in(a, 4), self.site_in(b, 4), split=True)

        self.assertEqual([graph.name for graph in products], ["A", "B"])
        self.assertEqual(self.sequences(products[0]), ["GGGG" + self.SITE + "TTTT"])
        self.assertEqual(self.sequences(products[1]), ["AAAA" + self.SITE + "CCCC"])

    def test_direct_repeats_in_one_molecule_excise_the_segment_as_a_circle(self):
        molecule = "GG" + self.SITE + "TACG" + self.SITE + "CC"
        graph = self.repository.import_sequence(molecule, name="line", sample="s")
        second = 2 + len(self.SITE) + 4

        products = graph.recombine(
            self.site_in(graph, 2), self.site_in(graph, second), split=True
        )

        self.assertEqual([product.name for product in products], ["line", "line-2"])
        self.assertEqual(self.sequences(products[0]), ["GG" + self.SITE + "CC"])
        self.assertEqual(self.sequences(products[1]), [self.RIGHT + "TACG" + self.LEFT])

    def test_an_excision_stays_in_one_graph_unless_split(self):
        molecule = "GG" + self.SITE + "TACG" + self.SITE + "CC"
        graph = self.repository.import_sequence(molecule, name="line", sample="s")
        second = 2 + len(self.SITE) + 4

        products = graph.recombine(self.site_in(graph, 2), self.site_in(graph, second))

        self.assertEqual([product.name for product in products], ["line"])
        self.assertEqual([item.name for item in self.repository.sample("s")], ["line"])
        self.assertEqual(self.sequences(graph), ["GG" + self.SITE + "CC"])
        # The excised circle is still in the graph, as a cycle no read from the start reaches.
        self.assertFalse(is_directed_acyclic_graph(graph.to_networkx()))

    def test_a_reversed_site_aligns_the_same_two_sites_as_inverted_repeats(self):
        molecule = "GG" + self.SITE + "TACG" + self.SITE + "CC"
        graph = self.repository.import_sequence(molecule, name="line", sample="s")
        second = 2 + len(self.SITE) + 4

        products = graph.recombine(
            self.site_in(graph, 2), self.site_in(graph, second).reverse_complement()
        )

        segment = self.RIGHT + "TACG" + self.LEFT
        self.assertEqual(len(products), 1)
        self.assertEqual(
            self.sequences(products[0]),
            ["GG" + self.LEFT + reverse_complement(segment) + self.RIGHT + "CC"],
        )

    def test_inverted_sites_invert_the_segment_whichever_site_is_on_the_reverse_strand(self):
        molecule = "GG" + reverse_complement(self.SITE) + "TACG" + self.SITE + "CC"
        second = 2 + len(self.SITE) + 4
        graph = self.repository.import_sequence(molecule, name="line", sample="s")
        mirror = self.repository.import_sequence(molecule, name="mirror", sample="m")

        inverted = graph.recombine(
            self.site_in(graph, 2).reverse_complement(), self.site_in(graph, second)
        )
        spelled_the_other_way = mirror.recombine(
            self.site_in(mirror, 2), self.site_in(mirror, second).reverse_complement()
        )

        self.assertEqual(self.sequences(inverted[0]), self.sequences(spelled_the_other_way[0]))
        self.assertNotEqual(self.sequences(inverted[0]), [molecule])

    def test_reversing_both_sites_is_the_same_exchange(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        b = self.repository.import_sequence("AAAA" + self.SITE + "TTTT", name="B", sample="s")

        products = a.recombine(
            self.site_in(a, 4).reverse_complement(), self.site_in(b, 4).reverse_complement()
        )

        sequences = sorted(sequence for graph in products for sequence in self.sequences(graph))
        self.assertEqual(
            sequences,
            sorted(["GGGG" + self.SITE + "TTTT", "AAAA" + self.SITE + "CCCC"]),
        )

    def test_inverted_sites_in_two_linear_molecules_are_not_supported_yet(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        b = self.repository.import_sequence("AAAA" + self.SITE + "TTTT", name="B", sample="s")

        with self.assertRaisesRegex(ValueError, "no start and the other no end"):
            a.recombine(self.site_in(a, 4), self.site_in(b, 4).reverse_complement())

    def test_inverted_sites_between_a_circle_and_a_line_integrate_it_reversed(self):
        line = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="line", sample="s")
        circle = self.repository.import_sequence(
            self.SITE + "TACG", name="circle", sample="s", circular=True
        )

        products = line.recombine(
            self.site_in(line, 4), self.site_in(circle, 0).reverse_complement()
        )

        self.assertEqual(len(products), 1)
        self.assertEqual(len(self.sequences(products[0])), 1)
        self.assertEqual(len(self.sequences(products[0])[0]), 4 + 34 + 34 + 4 + 4)

    def test_a_site_with_a_reversed_copy_has_the_same_cuts(self):
        graph = self.repository.import_sequence(self.SITE, name="s", sample="s")
        site = gen.Site(graph.locus(0, len(self.SITE)), 4, 9)

        reverse = site.reverse_complement()

        self.assertEqual(reverse.overhang, site.overhang)
        self.assertEqual(reverse.reverse_complement(), site)

    def test_sites_that_cut_with_different_overhangs_are_rejected(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        b = self.repository.import_sequence("AAAA" + self.SITE + "TTTT", name="B", sample="s")

        with self.assertRaisesRegex(ValueError, "overhangs of 2 and 0"):
            a.recombine(self.site_in(a, 4, top=15, bottom=17), self.site_in(b, 4))

    def test_sites_whose_overhangs_differ_in_their_bases_are_rejected(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        other = self.LEFT + "GC" + self.RIGHT[2:]
        b = self.repository.import_sequence("AAAA" + other + "TTTT", name="B", sample="s")

        with self.assertRaisesRegex(ValueError, "bubble"):
            a.recombine(
                self.site_in(a, 4, top=17, bottom=19), self.site_in(b, 4, top=17, bottom=19)
            )

    def test_staggered_sites_with_matching_overhangs_make_the_blunt_junction(self):
        a = self.repository.import_sequence("GGGG" + self.SITE + "CCCC", name="A", sample="s")
        b = self.repository.import_sequence("AAAA" + self.SITE + "TTTT", name="B", sample="s")

        products = a.recombine(
            self.site_in(a, 4, top=15, bottom=17),
            self.site_in(b, 4, top=15, bottom=17),
            split=True,
        )

        self.assertEqual(self.sequences(products[0]), ["GGGG" + self.SITE + "TTTT"])
        self.assertEqual(self.sequences(products[1]), ["AAAA" + self.SITE + "CCCC"])

    def test_recombine_needs_sites(self):
        a = self.repository.import_sequence("GGGG" + self.SITE, name="A", sample="s")

        with self.assertRaises(TypeError):
            a.recombine(a.locus(0, 4), a.locus(0, 4))

    def test_a_plasmid_integrates_into_a_chromosome_through_attp_and_attb(self):
        core = "TTGC"
        attb = "GCGTAACG" + core + "CAGTTCAG"
        attp = "ACCTGAAT" + core + "GGTCATCC"
        backbone = "TACGTACGAA"
        chromosome = self.repository.import_sequence(
            "CCCC" + attb + "GGGG", name="chromosome", sample="s"
        )
        plasmid = self.repository.import_sequence(
            attp + backbone, name="plasmid", sample="s", circular=True
        )
        # The core is cut one base in, leaving the two bases between the cuts as the overhang.
        b_site = gen.Site(chromosome.locus(4, 4 + len(attb)), 9, 11)
        p_site = gen.Site(plasmid.locus(0, len(attp)), 9, 11)

        products = chromosome.recombine(b_site, p_site)

        self.assertEqual([graph.name for graph in products], ["chromosome"])
        attl = "GCGTAACG" + core + "GGTCATCC"
        attr = "ACCTGAAT" + core + "CAGTTCAG"
        self.assertEqual(
            self.sequences(products[0]), ["CCCC" + attl + backbone + attr + "GGGG"]
        )

    def test_serine_recombination_needs_the_cores_to_match(self):
        attb = "GCGTAACG" + "TTGC" + "CAGTTCAG"
        attp = "ACCTGAAT" + "TTCC" + "GGTCATCC"
        chromosome = self.repository.import_sequence(
            "CCCC" + attb + "GGGG", name="chromosome", sample="s"
        )
        plasmid = self.repository.import_sequence(
            attp + "TACGTACGAA", name="plasmid", sample="s", circular=True
        )

        with self.assertRaisesRegex(ValueError, "bubble"):
            chromosome.recombine(
                gen.Site(chromosome.locus(4, 4 + len(attb)), 9, 11),
                gen.Site(plasmid.locus(0, len(attp)), 9, 11),
            )


class MotifTests(SitesTestCase):
    LOX = "ATAACTTCGTATAATGTATGCTATACGAAGTTAT"
    LOXPSYM = "ATAACTTCGTATATTATATAATATACGAAGTTAT"

    def test_a_motif_applied_to_a_locus_makes_a_site_with_its_cuts(self):
        motif = gen.Motif("AANCTT", 2, 4)
        graph = self.repository.import_sequence("GGAACCTTGG", name="m", sample="s")

        [hit] = graph.search(motif.recognition_site)
        site = motif.apply(hit)

        self.assertEqual(site.locus.sequence, "AACCTT")
        self.assertEqual((site.top, site.bottom), (2, 4))
        self.assertIs(site.motif, motif)

    def test_applying_a_motif_to_a_reverse_strand_hit_reads_along_the_motif(self):
        motif = gen.Motif(self.LOX, 17)
        graph = self.repository.import_sequence(
            "GG" + reverse_complement(self.LOX) + "CC", name="m", sample="s"
        )

        [hit] = graph.search(motif.recognition_site)
        site = motif.apply(hit)

        self.assertEqual(site.locus.sequence, self.LOX)
        self.assertEqual(site.locus.direction, "-")

    def test_applying_a_motif_to_the_wrong_bases_raises(self):
        graph = self.repository.import_sequence("GGGGGGGG", name="m", sample="s")

        with self.assertRaisesRegex(ValueError, "not an instance"):
            gen.Motif("AANNTT", 2).apply(graph.locus(0, 6))

    def test_a_motif_knows_whether_it_is_palindromic(self):
        self.assertTrue(gen.Motif(self.LOXPSYM, 17).palindromic)
        self.assertFalse(gen.Motif(self.LOX, 17).palindromic)

    def test_a_cut_outside_the_recognition_site_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside a recognition site"):
            gen.Motif("AANNTT", 7)

    def test_a_site_reads_the_parts_its_motif_names(self):
        motif = gen.Motif(
            self.LOX, 17, parts={"left": (0, 13), "spacer": (13, 21), "right": (21, 34)}
        )
        graph = self.repository.import_sequence("GG" + self.LOX + "CC", name="p", sample="s")

        [hit] = graph.search(motif.recognition_site)
        site = motif.apply(hit)

        self.assertEqual(site.part("left").sequence, self.LOX[:13])
        self.assertEqual(site.part("spacer").sequence, "ATGTATGC")
        self.assertEqual(site.part("right").sequence, self.LOX[21:])

    def test_a_part_of_a_reverse_strand_site_reads_along_the_motif(self):
        motif = gen.Motif(self.LOX, 17, parts={"spacer": (13, 21)})
        graph = self.repository.import_sequence(
            "GG" + reverse_complement(self.LOX) + "CC", name="p", sample="s"
        )

        [hit] = graph.search(motif.recognition_site)
        site = motif.apply(hit)

        self.assertEqual(site.part("spacer").sequence, "ATGTATGC")
        self.assertEqual(
            site.reverse_complement().part("spacer").sequence, reverse_complement("ATGTATGC")
        )

    def test_an_unnamed_part_raises(self):
        motif = gen.Motif("AANNTT", 2)
        graph = self.repository.import_sequence("GGAAGGTTGG", name="p", sample="s")
        site = motif.apply(graph.locus(2, 8))

        with self.assertRaisesRegex(KeyError, "no part named"):
            site.part("spacer")
        with self.assertRaisesRegex(KeyError, "no part named"):
            gen.Site(graph.locus(2, 8), 2).part("spacer")

    def test_a_part_outside_the_recognition_site_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a start and end"):
            gen.Motif("AANNTT", 2, parts={"arm": (4, 8)})

    def test_sites_with_different_overhangs_are_not_compatible(self):
        graph = self.repository.import_sequence("AAAATTTTGGGGCCCC", name="m", sample="s")

        first = gen.Site(graph.locus(0, 8), 2, 4)
        second = gen.Site(graph.locus(8, 16), 2, 4)
        third = gen.Site(graph.locus(8, 16), 2, 3)

        self.assertFalse(first.compatible(second))
        self.assertFalse(first.compatible(third))
        self.assertTrue(first.compatible(gen.Site(graph.locus(0, 8), 2, 4)))


class ClosureTests(SitesTestCase):
    LOX = "ATAACTTCGTATAATGTATGCTATACGAAGTTAT"
    LOXPSYM = "ATAACTTCGTATATTATATAATATACGAAGTTAT"

    PROMOTERS = ["TTGACACCTATA", "CGGATAGCTTAC"]

    def cassette(self, site, directions):
        parts = ["GCGGTTAC"]
        for index, direction in enumerate(directions):
            parts.append(site if direction == "+" else reverse_complement(site))
            if index < len(directions) - 1:
                parts.append(self.PROMOTERS[index])
        parts.append("TTCGACGA")
        return self.repository.import_sequence("".join(parts), name="c", sample="c")

    def test_the_pairs_of_a_palindromic_site_come_in_both_alignments_once(self):
        motif = gen.Motif(self.LOXPSYM, 17)
        graph = self.cassette(self.LOXPSYM, "++")
        sites = [motif.apply(hit) for hit in graph.search(motif.recognition_site)]

        self.assertEqual(len(sites), 4)
        self.assertEqual(len(list(gen.site_pairs(sites))), 2)

    def test_a_directional_pair_has_the_alignment_its_bases_fix(self):
        motif = gen.Motif(self.LOX, 17)
        graph = self.cassette(self.LOX, "++")
        hits = graph.search(self.LOX, sequence_kind="exact")

        self.assertEqual(len(list(gen.site_pairs([motif.apply(hit) for hit in hits]))), 1)

    def test_closure_of_two_direct_sites_holds_the_parent_and_the_excision(self):
        motif = gen.Motif(self.LOX, 17)
        graph = self.cassette(self.LOX, "++")

        tried, skipped = gen.recombination_closure(graph, motif)

        self.assertGreaterEqual(tried, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(graph.all_sequences()), 2)

    def test_closure_of_three_symmetric_sites_finds_every_layout(self):
        motif = gen.Motif(self.LOXPSYM, 17)
        graph = self.cassette(self.LOXPSYM, "+++")

        gen.recombination_closure(graph, motif)

        self.assertEqual(len(graph.all_sequences()), 13)

    def test_a_compatible_rule_narrows_the_pairs(self):
        motif = gen.Motif(self.LOX, 17)
        graph = self.cassette(self.LOX, "+++")

        tried, _ = gen.recombination_closure(graph, motif, compatible=lambda first, second: False)

        self.assertEqual(tried, 0)
        self.assertEqual(len(graph.all_sequences()), 1)


class AmbiguousBreakTests(SitesTestCase):
    def test_a_site_resolves_its_breakpoints_when_made(self):
        graph, base = self.line()

        site = gen.Site(graph.locus(2, 7), 3)

        self.assertEqual(tuple(site.top_break), (base(4), base(5)))
        self.assertEqual(site.breakpoints(), (site.top_break, site.bottom_break))

    def test_a_cut_beside_a_fork_raises_when_the_site_is_made(self):
        graph, base = self.line()
        graph.exchange((base(2), base(3)), (base(4), base(5)), stack=True)

        with self.assertRaises(gen.SuperpositionError):
            gen.Site(graph.locus(3, 5), 0)
