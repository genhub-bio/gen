"""The public names, types and hash handling that make the API predictable for agents."""

import gen

from test_api import FIXTURES, RepositoryTestCase


class PublicNamespaceTests(RepositoryTestCase):
    def test_dir_lists_only_the_public_names(self):
        self.assertEqual(set(dir(gen)), set(gen.__all__))

    def test_internal_classes_and_leaked_helpers_are_not_public(self):
        for name in (
            "NodeSlice",
            "SequencePart",
            "GraphWidget",
            "TextGraphWidget",
            "version",
            "PackageNotFoundError",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, gen.__all__)
                self.assertNotIn(name, dir(gen))

    def test_removed_and_hidden_repository_methods_are_not_listed(self):
        public = {name for name in dir(self.repository) if not name.startswith("_")}
        for name in (
            "plot",
            "make_stitch",
            "update_with_sequence",
            "get_sequence_graphs_by_collection",
            "get_sequence_graph_by_id",
            "get_node_sequence",
            "get_remotes",
            "db_path",
            "export_fasta",
            "export_gfa",
            "export_genbank",
            "derive_chunks",
            "derive_subgraph",
        ):
            with self.subTest(name=name):
                self.assertNotIn(name, public)


class HashHandlingTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_fasta(str(FIXTURES / "simple.fa"))[0]

    def test_sequence_graph_ids_are_hashable_and_round_trip(self):
        self.assertIsInstance(self.graph.id, gen.HashId)
        self.assertEqual(self.repository.get_sequence_graph(self.graph.id), self.graph)
        self.assertEqual(
            self.repository.get_sequence_graph(str(self.graph.id)), self.graph
        )
        self.assertEqual({self.graph.id: "graph"}[self.graph.id], "graph")

    def test_operation_ids_and_branch_heads_are_hash_ids(self):
        [operation, *_] = self.repository.get_operations()
        self.assertIsInstance(operation.id, gen.HashId)
        self.assertEqual(len(operation.id.to_bytes()), 20)
        self.assertEqual(self.repository.current_branch.head, operation.id)
        self.assertEqual(hash(self.repository.current_branch.head), hash(operation.id))
        self.assertEqual({operation: "head"}[self.repository.get_operations()[0]], "head")
        self.assertEqual(str(operation), str(operation.id))

    def test_operation_hash_ids_are_accepted_as_references(self):
        base = self.repository.get_operations()[0]
        branch = self.repository.create_branch("from_hash", base.id)
        self.assertEqual(branch.head, base.id)

    def test_annotation_ids_are_hash_ids_with_value_equality(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        annotation = self.graph.add_annotation(locus, "site", track="sites")
        [stored] = [item for item in self.graph.annotations if item.name == "site"]
        self.assertIsInstance(stored.id, gen.HashId)
        self.assertEqual(stored, annotation)
        self.assertEqual(hash(stored), hash(annotation))
        self.assertEqual(len({stored, annotation}), 1)
        self.assertNotEqual(stored, gen.Annotation(locus[1:4], "site"))

    def test_annotation_exposes_track_but_not_group_or_segments(self):
        [locus] = self.graph.search("GGAACACA", sequence_kind="exact")
        self.graph.add_annotation(locus, "site", track="sites")
        [stored] = [item for item in self.graph.annotations if item.name == "site"]
        self.assertEqual(stored.track, "sites")
        self.assertFalse(hasattr(stored, "group"))
        self.assertFalse(hasattr(stored, "segments"))

    def test_asset_ids_are_hash_ids(self):
        [asset, *_] = self.repository.get_assets()
        self.assertIsInstance(asset.id, gen.HashId)
        self.assertEqual({asset: 1}[self.repository.get_assets()[0]], 1)


class SampleAndGraphNamingTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        self.graph = self.sample[0]

    def test_sample_and_graph_report_collection_and_sample(self):
        self.assertEqual(self.graph.collection, self.sample.collection)
        self.assertIsInstance(self.graph.sample, gen.Sample)
        self.assertEqual(self.graph.sample.name, self.sample.name)
        self.assertEqual(list(self.graph.sample), list(self.sample))

    def test_get_sequence_graphs_filters(self):
        self.sample.copy("other")
        self.assertEqual(len(self.repository.get_sequence_graphs()), 2)
        by_name = self.repository.get_sequence_graphs(name=self.graph.name)
        self.assertEqual(len(by_name), 2)
        [only] = self.repository.get_sequence_graphs(sample="other")
        self.assertEqual(only.sample.name, "other")
        self.assertEqual(
            self.repository.get_sequence_graphs(sample=self.sample), [self.graph]
        )
        self.assertEqual(
            len(self.repository.get_sequence_graphs(collection=self.sample.collection)),
            2,
        )
        self.assertEqual(self.repository.get_sequence_graphs(collection="absent"), [])

    def test_remotes_is_a_property(self):
        self.assertEqual(self.repository.remotes, [])
        self.repository.add_remote("origin", "https://example.com/r")
        self.assertEqual([remote.name for remote in self.repository.remotes], ["origin"])


class NodeAndPositionTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_fasta(str(FIXTURES / "simple.fa"))[0]
        [self.locus] = self.graph.search("GGAACACA", sequence_kind="exact")

    def test_node_reads_its_own_sequence(self):
        node = self.locus.start().node
        self.assertEqual(len(node.sequence), node.length)
        self.assertIn("GGAACACA", node.sequence)
        for graph_node in self.graph.to_dict()["nodes"]:
            self.assertEqual(len(graph_node.sequence), graph_node.length)

    def test_position_and_superposition_report_their_graph(self):
        position = self.locus.start()
        self.assertEqual(position.graph, self.graph)
        self.assertEqual((position | self.locus.end()).graph, self.graph)
        self.assertFalse(hasattr(position, "sequence_graph"))

    def test_on_takes_a_graph(self):
        other = self.graph.sample.copy("other")[0]
        self.assertEqual(self.locus.start().on(graph=other).graph, other)

    def test_locus_has_no_slices(self):
        self.assertFalse(hasattr(self.locus, "slices"))


class SequenceTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence("ACGTACGT", name="seq", sample="s")

    def test_all_sequences_yield_sequence_objects_that_read_like_strings(self):
        [sequence] = list(self.graph.all_sequences())
        self.assertIsInstance(sequence, gen.Sequence)
        self.assertEqual(str(sequence), "ACGTACGT")
        self.assertEqual(sequence, "ACGTACGT")
        self.assertEqual(len(sequence), 8)
        self.assertEqual(sequence[1], "C")
        self.assertEqual(str(sequence[2:5]), "GTA")
        self.assertIn("GTAC", sequence)
        self.assertEqual({sequence: 1}["ACGTACGT"], 1)
        self.assertIsNone(sequence.name)
        self.assertLess(gen.Sequence("a", "AAA"), sequence)
        self.assertEqual(
            [str(item) for item in sorted([sequence, gen.Sequence(None, "AAA"), "CC"])],
            ["AAA", "ACGTACGT", "CC"],
        )
        with self.assertRaises(TypeError):
            sequence < 3

    def test_named_sequences_build_libraries(self):
        part = gen.Sequence
        graph = self.repository.import_library(
            "lib", [[part("a", "AAAA")], [part("b", "CC"), part("c", "GG")]]
        )
        self.assertEqual(
            sorted(str(sequence) for sequence in graph.all_sequences()),
            ["AAAACC", "AAAAGG"],
        )

    def test_unnamed_sequences_cannot_be_library_parts(self):
        [sequence] = list(self.graph.all_sequences())
        with self.assertRaises(ValueError):
            self.repository.import_library("lib", [[sequence]])


class AssetTests(RepositoryTestCase):
    def test_add_file_keeps_a_file_without_importing_sequence(self):
        readme = self.root / "README.txt"
        readme.write_text("notes for the design")
        asset = self.repository.add_file(readme, message="add readme")
        self.assertIsInstance(asset, gen.Asset)
        self.assertEqual(asset.name, "README.txt")
        self.assertEqual(self.repository.get_operations()[0].message, "add readme")
        self.assertEqual(self.repository.get_sequence_graphs(), [])
        self.assertIn(asset, self.repository.get_assets())

    def test_asset_path_is_the_hashed_stored_copy(self):
        readme = self.root / "README.txt"
        readme.write_text("notes for the design")
        asset = self.repository.add_file(str(readme))
        self.assertTrue(asset.path.is_file())
        self.assertNotEqual(asset.path.name, "README.txt")
        self.assertEqual(asset.path.read_text(), "notes for the design")
        self.assertIn(self.repository.gen_dir, asset.path.parents)

    def test_save_as_copies_under_a_chosen_name(self):
        readme = self.root / "README.txt"
        readme.write_text("notes for the design")
        asset = self.repository.add_file(readme)
        copy = asset.save_as(self.root / "renamed.md")
        self.assertEqual(copy.read_text(), "notes for the design")
        folder = self.root / "out"
        folder.mkdir()
        inside = asset.save_as(folder)
        self.assertEqual(inside.name, "README.txt")
        with self.assertRaises(FileExistsError):
            asset.save_as(self.root / "renamed.md")
        readme.unlink()
        self.assertEqual(asset.save_as(self.root / "renamed.md", overwrite=True), copy)

    def test_add_file_rejects_duplicates_and_missing_files(self):
        readme = self.root / "README.txt"
        readme.write_text("notes")
        self.repository.add_file(readme)
        with self.assertRaises(RuntimeError):
            self.repository.add_file(readme)
        with self.assertRaises(FileNotFoundError):
            self.repository.add_file(self.root / "absent.txt")

    def test_imported_files_are_assets_too(self):
        self.repository.import_fasta(str(FIXTURES / "simple.fa"))
        [asset] = self.repository.get_assets()
        self.assertEqual(asset.name, "simple.fa")
        self.assertIn(">", asset.path.read_text())


class StitchTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.left = self.repository.import_sequence("AAAACCCCGGGGTTTT", name="left", sample="s")
        self.right = self.repository.import_sequence("ACGTACGTAC", name="right", sample="s")

    def sequences(self, graph):
        return sorted(str(sequence) for sequence in graph.all_sequences())

    def test_stitching_graphs_concatenates_them(self):
        stitched = self.repository.stitch([self.left, self.right], "joined", "both")
        self.assertEqual(self.sequences(stitched), ["AAAACCCCGGGGTTTTACGTACGTAC"])
        self.assertEqual(stitched.sample.name, "joined")

    def test_stitching_loci_does_not_need_subgraphs(self):
        before = len(self.repository.get_sequence_graphs())
        stitched = self.repository.stitch(
            [self.left.region("left:4-8"), self.right.region("right:2-6")],
            "joined",
            "pieces",
        )
        self.assertEqual(self.sequences(stitched), ["CCCCGTAC"])
        self.assertEqual(len(self.repository.get_sequence_graphs()), before + 1)

    def test_a_graph_and_a_locus_can_be_mixed(self):
        stitched = self.repository.stitch(
            [self.left.region("left:0-4"), self.right], "joined", "mixed"
        )
        self.assertEqual(self.sequences(stitched), ["AAAAACGTACGTAC"])

    def test_stitched_locus_keeps_every_variant_inside_it(self):
        variant = self.left.sample.copy("variant")[0]
        variant.replace("left:6-8", "TT", stack=True)
        stitched = self.repository.stitch(
            [variant.region("left:4-12"), self.right.region("right:0-2")],
            "joined",
            "variants",
        )
        self.assertEqual(
            self.sequences(stitched), ["CCCCGGGGAC", "CCTTGGGGAC"]
        )

    def test_stitching_a_multi_node_locus_keeps_a_current_path(self):
        edited = self.left.sample.copy("edited")[0]
        edited.replace("left:6-8", "TT")
        stitched = self.repository.stitch(
            [edited.region("left:4-12"), self.right.region("right:0-2")], "joined", "edited"
        )
        self.assertEqual(self.sequences(stitched), ["CCTTGGGGAC"])
        path = self.root / "stitched.fa"
        stitched.export_fasta(str(path))
        self.assertIn("CCTTGGGGAC", path.read_text())

    def test_stitching_rejects_bad_parts(self):
        with self.assertRaises(ValueError):
            self.repository.stitch([], "joined", "none")
        with self.assertRaises(TypeError):
            self.repository.stitch([self.left, "right"], "joined", "bad")
        with self.assertRaises(ValueError):
            self.repository.stitch(
                [self.left.region("left:0-4").reverse_complement(), self.right],
                "joined",
                "reversed",
            )
        with self.assertRaises(RuntimeError):
            self.repository.stitch([self.left, self.left], "joined", "twice")


class SubgraphAndCoordinateTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence(
            "AAAACCCCGGGGTTTT", name="seq", sample="s"
        )
        self.copy = self.graph.sample.copy("copy")[0]

    def test_graph_locus_matches_the_region_string(self):
        self.assertEqual(self.graph.locus(4, 8), self.graph.region("seq:4-8"))
        self.assertEqual(self.graph.locus(4, 8).sequence, "CCCC")
        for start, end in ((-1, 3), (5, 5), (8, 4)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    self.graph.locus(start, end)
        with self.assertRaises(ValueError):
            self.graph.locus(10, 99)

    def sequences(self, graph):
        return sorted(str(sequence) for sequence in graph.all_sequences())

    def test_subgraph_from_a_locus_uses_its_start_and_end(self):
        child = self.graph.subgraph("sub", self.graph.locus(4, 12))
        self.assertEqual(self.sequences(child), ["CCCCGGGG"])
        self.assertEqual(child.sample.name, "sub")

    def test_subgraph_from_a_locus_of_another_graph(self):
        locus = self.graph.locus(2, 10)
        self.copy.replace("seq:5-7", "TTT")
        child = self.copy.subgraph("sub", locus)
        self.assertEqual(self.sequences(child), ["AACTTTCGG"])

    def test_subgraph_from_positions_keeps_every_variant(self):
        locus = self.graph.locus(2, 14)
        variant = self.copy
        variant.replace("seq:6-8", "TT", stack=True)
        child = variant.subgraph("sub", locus.start(), locus.end())
        self.assertEqual(
            self.sequences(child), ["AACCCCGGGGTT", "AACCTTGGGGTT"]
        )

    def test_subgraph_by_coordinates_still_works(self):
        self.assertEqual(self.sequences(self.graph.subgraph("sub", 4, 8)), ["CCCC"])

    def test_subgraph_from_a_reverse_locus(self):
        child = self.graph.subgraph("sub", self.graph.locus(4, 12).reverse_complement())
        self.assertEqual(self.sequences(child), ["CCCCGGGG"])

    def test_subgraph_fails_when_a_position_is_gone(self):
        locus = self.graph.locus(2, 10)
        self.copy.delete("seq:8-12")
        with self.assertRaises(ValueError):
            self.copy.subgraph("sub", locus)

    def test_subgraph_fails_when_the_end_cannot_be_reached(self):
        locus = self.graph.locus(2, 10)
        with self.assertRaises(ValueError):
            self.graph.subgraph("sub", locus.end(), locus.start())
        other = self.repository.import_sequence("TTTTTTTT", name="other", sample="o")
        with self.assertRaises(ValueError):
            other.subgraph("sub", locus)

    def test_subgraph_rejects_other_argument_shapes(self):
        locus = self.graph.locus(2, 10)
        with self.assertRaises(TypeError):
            self.graph.subgraph("sub", locus, 4)
        with self.assertRaises(TypeError):
            self.graph.subgraph("sub", "seq:2-10")

    def test_chunks_come_back_in_order_along_the_graph(self):
        chunks = self.graph.chunks("pieces", chunk_size=1)

        self.assertEqual(len(chunks), 16)
        self.assertEqual(
            "".join(self.sequences(chunk)[0] for chunk in chunks), "AAAACCCCGGGGTTTT"
        )
        stitched = self.repository.stitch(chunks, "rebuilt", "whole")
        self.assertEqual(self.sequences(stitched), ["AAAACCCCGGGGTTTT"])

    def test_locus_has_no_on_method(self):
        self.assertFalse(hasattr(self.graph.locus(2, 10), "on"))

    def pathless_subgraph(self, new_sample):
        """A subgraph from a position off the current path, so it has no path of its own."""
        variant = self.graph.sample.copy(new_sample + "_source")[0]
        inserted = variant.insert("GG", after=variant.locus(4, 6).start(), stack=True)
        return variant.subgraph(new_sample, inserted.start(), variant.locus(8, 10).end())

    def test_a_graph_without_a_current_path_cannot_be_stitched(self):
        pathless = self.pathless_subgraph("loose")
        self.assertEqual(self.sequences(pathless), ["GGCCCGG"])
        self.assertIsNone(self.export_path(pathless))
        last = self.graph.subgraph("last", self.graph.locus(12, 16))
        for parts in ([pathless], [last, pathless], [pathless, last]):
            with self.subTest(order=[part.sample.name for part in parts]):
                with self.assertRaises(RuntimeError) as raised:
                    self.repository.stitch(parts, "joined", "none")
                self.assertIn("no current path", str(raised.exception))

    def test_a_locus_of_a_pathless_graph_can_still_be_stitched(self):
        pathless = self.pathless_subgraph("loose")
        [piece] = pathless.search("GGCC", sequence_kind="exact")
        stitched = self.repository.stitch(
            [piece, self.graph.locus(12, 16)], "joined", "locus"
        )
        self.assertEqual(self.sequences(stitched), ["GGCCTTTT"])

    def export_path(self, graph):
        """The exported FASTA sequence, or None when the graph has no current path."""
        out = self.root / (graph.sample.name + ".fa")
        try:
            graph.export_fasta(str(out))
        except RuntimeError as error:
            self.assertIn("No current path", str(error))
            return None
        return "".join(
            line for line in out.read_text().splitlines() if not line.startswith(">")
        )
