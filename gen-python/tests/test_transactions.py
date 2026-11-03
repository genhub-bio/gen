"""Trial repository edits can be undone or committed as one history operation."""

import gc
import gen

from test_api import FIXTURES, RepositoryTestCase


class TransactionTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.graph = self.repository.import_sequence(
            "AACCGG", name="vector", sample="parent"
        )
        self.sample = self.graph.sample
        self.base_operation = self.repository.get_operations()[0]
        self.base_operation_count = len(self.repository.get_operations())

    def sequence(self, graph=None):
        graph = graph or self.graph
        [sequence] = graph.all_sequences()
        return str(sequence)

    def test_edits_still_commit_immediately_outside_a_transaction(self):
        self.graph.replace("vector:1-3", "TT", message="replace motif")

        self.assertEqual(self.sequence(), "ATTCGG")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )
        self.assertEqual(self.repository.get_operations()[0].message, "replace motif")

    def test_transaction_factory_does_not_start_a_sql_savepoint(self):
        transaction = self.repository.transaction()
        self.graph.replace("vector:1-3", "TT", message="outside entered context")

        self.assertEqual(self.sequence(), "ATTCGG")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )
        del transaction
        gc.collect()

        self.assertEqual(self.sequence(), "ATTCGG")
        self.assertEqual(
            self.repository.get_operations()[0].message, "outside entered context"
        )

    def test_raw_sql_transactions_control_writes_and_asset_reads(self):
        notes = self.root / "raw-transaction-notes.txt"
        notes.write_text("pending raw transaction asset\n")
        head = self.repository.current_branch.head
        operation_count = len(self.repository.get_operations())

        self.repository.execute("BEGIN")
        self.graph.replace("vector:1-3", "TT")
        self.assertEqual(self.sequence(), "ATTCGG")
        self.assertEqual(self.repository.current_branch.head, head)
        self.repository.execute("ROLLBACK")

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, head)

        self.repository.execute("BEGIN")
        asset = self.repository.add_file(notes)
        self.assertEqual(asset.name, "raw-transaction-notes.txt")
        self.assertIn(
            "raw-transaction-notes.txt",
            {item.name for item in self.repository.get_assets()},
        )
        self.assertNotIn(
            "raw-transaction-notes.txt",
            {
                item.name
                for item in self.repository.get_assets(
                    branch=self.repository.current_branch
                )
            },
        )
        self.repository.execute("COMMIT")

        self.assertEqual(self.repository.current_branch.head, head)
        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertIn(
            "raw-transaction-notes.txt",
            {item.name for item in self.repository.get_assets()},
            "default asset reads should include Dolt's current working set",
        )
        self.assertNotIn(
            "raw-transaction-notes.txt",
            {
                item.name
                for item in self.repository.get_assets(
                    branch=self.repository.current_branch
                )
            },
            "explicit branch reads should show committed assets only",
        )

        self.graph.replace("vector:1-3", "TT", message="commit working set")
        self.assertEqual(len(self.repository.get_operations()), operation_count + 1)
        self.assertEqual(
            self.repository.get_operations()[0].message, "commit working set"
        )

    def test_transaction_inside_raw_transaction_releases_into_parent(self):
        head = self.repository.current_branch.head
        operation_count = len(self.repository.get_operations())

        self.repository.execute("BEGIN")
        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            self.assertIsNone(transaction.commit("inner savepoint"))

        self.assertEqual(self.sequence(), "ATTCGG")
        self.assertEqual(self.repository.current_branch.head, head)
        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.repository.execute("ROLLBACK")

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, head)
        self.assertEqual(len(self.repository.get_operations()), operation_count)

    def test_undo_stack_and_commit_create_one_persisted_operation(self):
        graph_handle = self.graph
        sample_handle = self.sample
        isolated_reader = gen.Repository(str(self.root / "repository"))
        [committed_view] = isolated_reader.get_sequence_graphs(
            name="vector", sample="parent"
        )

        with self.repository.transaction() as transaction:
            graph_handle.replace("vector:1-3", "TT")
            graph_handle.replace("vector:4-6", "AA")

            self.assertEqual(self.sequence(graph_handle), "ATTCAA")
            [pending_graph] = self.repository.get_sequence_graphs(
                name="vector", sample="parent"
            )
            self.assertEqual(self.sequence(pending_graph), "ATTCAA")
            self.assertEqual(self.sequence(committed_view), "AACCGG")
            self.assertEqual(sample_handle.metadata, {})
            self.assertEqual(
                self.repository.current_branch.head, self.base_operation.id
            )
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count
            )

            self.assertTrue(transaction.undo())
            self.assertEqual(self.sequence(graph_handle), "ATTCGG")
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sequence(graph_handle), "AACCGG")
            self.assertFalse(transaction.undo())

            graph_handle.replace("vector:1-3", "TT")
            committed = transaction.commit("trial accepted")
            self.assertIsNotNone(committed)
            self.assertEqual(committed.message, "trial accepted")
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count + 1
            )
            self.assertEqual(
                self.repository.get_operations()[0].message, "trial accepted"
            )
            self.assertIsInstance(self.repository.get_assets(), list)

        reopened = gen.Repository(str(self.root / "repository"))
        [reopened_graph] = reopened.get_sequence_graphs(name="vector", sample="parent")
        self.assertEqual(self.sequence(reopened_graph), "ATTCGG")
        self.assertEqual(reopened.current_branch.head, committed.id)

    def test_repeated_undo_restores_each_prior_sequence_state(self):
        sequence_a = "AACCGG"
        sequence_b = "ATTCGG"
        sequence_c = "ATTCAA"
        sequence_d = "ACCCAA"
        head_before = self.repository.current_branch.head
        operations_before = [
            (operation.id, operation.message)
            for operation in self.repository.get_operations()
        ]

        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            self.assertEqual(self.sequence(), sequence_b)
            self.graph.replace("vector:4-6", "AA")
            self.assertEqual(self.sequence(), sequence_c)
            self.graph.replace("vector:1-3", "CC")
            self.assertEqual(self.sequence(), sequence_d)

            for expected_sequence in (sequence_c, sequence_b, sequence_a):
                self.assertTrue(transaction.undo())
                self.assertEqual(self.sequence(), expected_sequence)
                self.assertEqual(self.repository.current_branch.head, head_before)
                self.assertEqual(
                    [
                        (operation.id, operation.message)
                        for operation in self.repository.get_operations()
                    ],
                    operations_before,
                )

            self.assertFalse(transaction.undo())
            self.assertEqual(self.sequence(), sequence_a)
            self.assertEqual(self.repository.current_branch.head, head_before)
            self.assertEqual(
                [
                    (operation.id, operation.message)
                    for operation in self.repository.get_operations()
                ],
                operations_before,
            )

        self.assertEqual(self.sequence(), sequence_a)

    def test_normal_context_exit_rolls_back_uncommitted_edits(self):
        with self.repository.transaction():
            self.graph.replace("vector:1-3", "TT")
            self.assertEqual(self.sequence(), "ATTCGG")

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, self.base_operation.id)
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count
        )

    def test_exception_context_exit_rolls_back_uncommitted_edits(self):
        with self.assertRaisesRegex(ValueError, "stop trial"):
            with self.repository.transaction():
                self.graph.replace("vector:1-3", "TT")
                raise ValueError("stop trial")

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, self.base_operation.id)
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count
        )

    def test_failed_multi_record_import_keeps_only_earlier_trial_edits(self):
        fasta = self.root / "partial-import.fa"
        fasta.write_bytes(b">temporary\nGATTACA\n>\xff\nTTTT\n")
        assets_before = {
            (asset.name, asset.id) for asset in self.repository.get_assets()
        }

        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.assertRaises(RuntimeError):
                self.repository.import_fasta(str(fasta), sample="parent")

            self.assertEqual(self.sequence(), "ATTCGG")
            self.assertEqual(
                self.repository.get_sequence_graphs(name="temporary", sample="parent"),
                [],
            )
            self.assertEqual(
                {(asset.name, asset.id) for asset in self.repository.get_assets()},
                assets_before,
            )
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count
            )
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sequence(), "AACCGG")
            self.assertFalse(transaction.undo())

    def test_graph_annotations_and_sample_metadata_can_be_undone(self):
        [locus] = self.graph.search("AAC", sequence_kind="exact")

        with self.repository.transaction() as transaction:
            self.sample.add_metadata({"purpose": "trial"})
            self.graph.add_annotation(locus, "trial-site", track="trial-sites")

            self.assertEqual(self.sample.metadata, {"purpose": "trial"})
            self.assertIn("trial-site", {item.name for item in self.graph.annotations})
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count
            )

            self.assertTrue(transaction.undo())
            self.assertNotIn(
                "trial-site", {item.name for item in self.graph.annotations}
            )
            self.assertEqual(self.sample.metadata, {"purpose": "trial"})
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sample.metadata, {})
            self.assertFalse(transaction.undo())

    def test_no_op_calls_do_not_add_undo_entries_and_empty_commit_returns_none(self):
        with self.repository.transaction() as transaction:
            self.repository.import_sequence(
                "AACCGG", name="vector", sample="parent", exist_ok=True
            )
            self.assertFalse(transaction.undo())

            self.sample.add_metadata({"purpose": "trial"})
            self.sample.add_metadata({"purpose": "trial"})
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sample.metadata, {})
            self.assertFalse(transaction.undo())

        with self.repository.transaction() as empty_transaction:
            self.assertIsNone(empty_transaction.commit("empty transaction"))

        self.assertEqual(self.repository.current_branch.head, self.base_operation.id)
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count
        )

    def test_vcf_updates_are_visible_undoable_and_committed_once_in_a_transaction(self):
        repository = gen.Repository(str(self.root / "vcf-repository"))
        repository.import_reference_fasta(str(FIXTURES / "simple.fa"), "reference")
        base_operation = repository.get_operations()[0]
        base_operation_count = len(repository.get_operations())
        vcf = str(FIXTURES / "simple.vcf")

        with repository.transaction() as transaction:
            samples = repository.update_with_vcf(vcf, reference="reference")
            self.assertEqual(
                sorted(sample.name for sample in samples), ["G1", "foo", "unknown"]
            )
            [foo_graph] = repository.get_sequence_graphs(name="m123", sample="foo")
            self.assertEqual(
                self.sequence(foo_graph), "ATCATCGATCGATCGATCGGGAACACACAGAGA"
            )
            self.assertEqual(repository.current_branch.head, base_operation.id)
            self.assertEqual(len(repository.get_operations()), base_operation_count)

            self.assertTrue(transaction.undo())
            self.assertEqual(repository.get_sequence_graphs(sample="foo"), [])
            self.assertEqual(repository.current_branch.head, base_operation.id)

            repository.update_with_vcf(vcf, reference="reference")
            committed = transaction.commit("apply VCF trial")
            self.assertIsNotNone(committed)
            self.assertEqual(committed.message, "apply VCF trial")

        reopened = gen.Repository(str(self.root / "vcf-repository"))
        [foo_graph] = reopened.get_sequence_graphs(name="m123", sample="foo")
        self.assertEqual(self.sequence(foo_graph), "ATCATCGATCGATCGATCGGGAACACACAGAGA")
        self.assertEqual(reopened.get_operations()[0].message, "apply VCF trial")
        self.assertEqual(len(reopened.get_operations()), base_operation_count + 1)

    def test_translation_is_visible_and_undoable_without_committing(self):
        dna_graph = self.repository.import_sequence(
            "TAAGGGCCC", name="gene", sample="translation"
        )
        operation_count = len(self.repository.get_operations())
        head = self.repository.current_branch.head

        with self.repository.transaction() as transaction:
            protein_graph = dna_graph.translate_annotation(name="gene protein")
            self.assertEqual(self.sequence(protein_graph), "*")
            [pending_protein] = self.repository.get_sequence_graphs(
                name="gene protein", sample="translation"
            )
            self.assertEqual(self.sequence(pending_protein), "*")
            self.assertEqual(len(self.repository.get_operations()), operation_count)
            self.assertEqual(self.repository.current_branch.head, head)

            self.assertTrue(transaction.undo())
            self.assertEqual(
                self.repository.get_sequence_graphs(
                    name="gene protein", sample="translation"
                ),
                [],
            )
            self.assertFalse(transaction.undo())

        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertEqual(self.repository.current_branch.head, head)

    def test_file_and_annotation_imports_are_staged_and_rolled_back(self):
        sample = self.repository.import_fasta(
            str(FIXTURES / "simple.fa"), sample="annotation-sample"
        )
        graph = sample[0]
        notes = self.root / "notes.txt"
        notes.write_text("trial file\n")
        operation_count = len(self.repository.get_operations())
        head = self.repository.current_branch.head

        with self.repository.transaction():
            asset = self.repository.add_file(notes)
            annotation_result = self.repository.import_annotations(
                str(FIXTURES / "simple.gff"), name="trial-genes"
            )

            self.assertIsNone(annotation_result)
            self.assertEqual(asset.name, "notes.txt")
            asset_names = {item.name for item in self.repository.get_assets()}
            self.assertIn("notes.txt", asset_names)
            self.assertIn("trial-genes", asset_names)
            self.assertIn("foobar", {item.name for item in graph.annotations})
            self.assertEqual(len(self.repository.get_operations()), operation_count)
            self.assertEqual(self.repository.current_branch.head, head)

        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertEqual(self.repository.current_branch.head, head)
        self.assertNotIn(
            "notes.txt", {item.name for item in self.repository.get_assets()}
        )
        self.assertNotIn(
            "trial-genes", {item.name for item in self.repository.get_assets()}
        )
        self.assertNotIn("foobar", {item.name for item in graph.annotations})

    def test_add_file_duplicate_keeps_its_error_and_preserves_trial_edits(self):
        notes = self.root / "notes.txt"
        notes.write_text("already committed\n")
        self.repository.add_file(notes)
        operation_count = len(self.repository.get_operations())
        head = self.repository.current_branch.head

        with self.assertRaisesRegex(RuntimeError, "already in the repository"):
            self.repository.add_file(notes)

        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.assertRaisesRegex(RuntimeError, "already in the repository"):
                self.repository.add_file(notes)

            self.assertEqual(self.sequence(), "ATTCGG")
            self.assertEqual(len(self.repository.get_operations()), operation_count)
            self.assertEqual(self.repository.current_branch.head, head)
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sequence(), "AACCGG")
            self.assertFalse(transaction.undo())

        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertEqual(self.repository.current_branch.head, head)

    def test_annotation_file_duplicate_preserves_trial_edits(self):
        annotation_file = self.root / "empty.gff3"
        annotation_file.write_text("##gff-version 3\n")
        self.repository.import_annotations(str(annotation_file), name="genes")
        operation_count = len(self.repository.get_operations())
        head = self.repository.current_branch.head
        base_sequence = self.sequence()

        with self.assertRaisesRegex(RuntimeError, "No Changes"):
            self.repository.import_annotations(str(annotation_file), name="genes")

        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            trial_sequence = self.sequence()
            self.assertNotEqual(trial_sequence, base_sequence)

            with self.assertRaisesRegex(RuntimeError, "No Changes"):
                self.repository.import_annotations(str(annotation_file), name="genes")

            self.assertEqual(self.sequence(), trial_sequence)
            self.assertEqual(len(self.repository.get_operations()), operation_count)
            self.assertEqual(self.repository.current_branch.head, head)
            self.assertTrue(transaction.undo())
            self.assertEqual(self.sequence(), base_sequence)
            self.assertFalse(transaction.undo())

        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertEqual(self.repository.current_branch.head, head)

    def test_transaction_commit_keeps_latest_content_for_a_replaced_asset_path(self):
        notes = self.root / "repository" / "notes.txt"
        notes.write_text("second trial\n")

        with self.repository.transaction() as transaction:
            self.repository.add_file(notes)
            notes.write_text("first trial\n")
            self.repository.add_file(notes)

            [preview_asset] = [
                asset
                for asset in self.repository.get_assets()
                if asset.name == "notes.txt"
            ]
            preview = preview_asset.save_as(self.root / "preview-notes.txt")
            self.assertEqual(preview.read_text(), "first trial\n")
            transaction.commit("replace notes twice")

        reopened = gen.Repository(str(self.root / "repository"))
        [committed_asset] = [
            asset for asset in reopened.get_assets() if asset.name == "notes.txt"
        ]
        saved = committed_asset.save_as(self.root / "committed-notes.txt")
        self.assertEqual(saved.read_text(), "first trial\n")

    def test_history_and_remote_changes_follow_open_sql_transaction(self):
        self.repository.create_branch("feature")
        branch_names = {branch.name for branch in self.repository.get_branches()}
        operation_count = len(self.repository.get_operations())
        remote_names = {remote.name for remote in self.repository.remotes}

        with self.repository.transaction():
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.create_branch("blocked")
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.checkout("blocked", create=True)
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.delete_branch("feature")
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.merge("feature")
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.reset(self.base_operation)
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.apply(self.base_operation)
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.add_remote("origin", "file:///tmp/origin")
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.set_default_remote()
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.set_branch_remote()
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.push()
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.pull()
            with self.assertRaisesRegex(RuntimeError, "SQLite transaction"):
                self.repository.fetch()
            self.assertEqual(self.repository.query("SELECT 1"), [[1]])
            self.repository.execute("CREATE TABLE raw_transaction_table (value TEXT)")
            self.assertEqual(
                self.repository.query(
                    "SELECT name FROM sqlite_master WHERE name = 'raw_transaction_table'"
                ),
                [["raw_transaction_table"]],
            )

        self.assertEqual(
            self.repository.query(
                "SELECT name FROM sqlite_master WHERE name = 'raw_transaction_table'"
            ),
            [],
            "the outer transaction should roll back raw SQL writes too",
        )
        self.assertEqual(
            {branch.name for branch in self.repository.get_branches()}, branch_names
        )
        self.assertEqual(len(self.repository.get_operations()), operation_count)
        self.assertEqual(
            {remote.name for remote in self.repository.remotes}, remote_names
        )

    def test_inner_transaction_commit_is_adopted_by_outer_commit(self):
        with self.repository.transaction() as outer_transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.repository.transaction() as inner_transaction:
                self.graph.replace("vector:4-6", "AA")
                self.assertIsNone(inner_transaction.commit("inner message"))

            self.assertEqual(self.sequence(), "ATTCAA")
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count
            )
            committed = outer_transaction.commit("outer batch")
            self.assertIsNotNone(committed)
            self.assertEqual(committed.message, "outer batch")

            self.graph.replace("vector:1-3", "GG", message="post commit write")
            self.assertEqual(self.sequence(), "AGGCAA")
            self.assertEqual(
                len(self.repository.get_operations()), self.base_operation_count + 2
            )

        operations = self.repository.get_operations()
        self.assertEqual(operations[0].message, "post commit write")
        self.assertEqual(operations[1].message, "outer batch")

    def test_inner_rollback_preserves_outer_edits(self):
        with self.repository.transaction() as outer_transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.repository.transaction() as inner_transaction:
                self.graph.replace("vector:4-6", "AA")
                inner_transaction.rollback()

            self.assertEqual(self.sequence(), "ATTCGG")
            self.graph.replace("vector:4-6", "CC")
            committed = outer_transaction.commit("outer edit after inner rollback")

        self.assertEqual(self.sequence(), "ATTCCC")
        self.assertEqual(committed.message, "outer edit after inner rollback")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )

    def test_three_nested_transactions_rollback_inner_and_commit_outer(self):
        with self.repository.transaction() as outer_transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.repository.transaction() as middle_transaction:
                self.graph.replace("vector:4-6", "AA")
                with self.repository.transaction():
                    self.graph.replace("vector:1-3", "CC")
                self.assertEqual(self.sequence(), "ATTCAA")
                self.assertIsNone(middle_transaction.commit("middle transaction"))
            committed = outer_transaction.commit("three-level transaction")

        self.assertEqual(self.sequence(), "ATTCAA")
        self.assertEqual(committed.message, "three-level transaction")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )

    def test_inner_exception_rolls_back_only_inner_edits(self):
        with self.repository.transaction() as outer_transaction:
            self.graph.replace("vector:1-3", "TT")
            with self.assertRaisesRegex(ValueError, "stop inner trial"):
                with self.repository.transaction():
                    self.graph.replace("vector:4-6", "AA")
                    raise ValueError("stop inner trial")

            self.assertEqual(self.sequence(), "ATTCGG")
            self.graph.replace("vector:4-6", "CC")
            committed = outer_transaction.commit("outer edit after inner exception")

        self.assertEqual(self.sequence(), "ATTCCC")
        self.assertEqual(committed.message, "outer edit after inner exception")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )

    def test_undo_crossing_inner_boundary_reports_missing_savepoint(self):
        with self.repository.transaction() as outer_transaction:
            self.graph.replace("vector:1-3", "TT")
            self.graph.replace("vector:4-6", "AA")
            with self.assertRaisesRegex(
                RuntimeError, "no such savepoint: gen_transaction"
            ):
                with self.repository.transaction() as inner_transaction:
                    self.graph.replace("vector:1-3", "CC")
                    self.assertTrue(inner_transaction.undo())
                    self.assertTrue(inner_transaction.undo())

            self.assertEqual(self.sequence(), "ATTCGG")
            self.graph.replace("vector:4-6", "CC")
            committed = outer_transaction.commit(
                "outer survives invalid inner savepoint"
            )

        self.assertEqual(self.sequence(), "ATTCCC")
        self.assertEqual(committed.message, "outer survives invalid inner savepoint")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )

    def test_writes_after_explicit_rollback_use_autocommit_before_context_exit(self):
        with self.repository.transaction() as transaction:
            self.graph.replace("vector:1-3", "TT")
            transaction.rollback()

            with self.assertRaises(RuntimeError):
                transaction.undo()
            self.graph.replace("vector:4-6", "AA", message="write after rollback")
            self.repository.create_branch("after-rollback")
            self.repository.add_remote("after-rollback", "file:///tmp/after-rollback")
            self.repository.set_default_remote("after-rollback")
            self.repository.set_branch_remote("after-rollback")
            self.assertIsInstance(self.repository.get_assets(), list)

        self.assertEqual(self.sequence(), "AACCAA")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )

    def test_open_transaction_drop_rolls_back_its_trial_edits(self):
        transaction = self.repository.transaction()
        transaction.__enter__()
        self.graph.replace("vector:1-3", "TT")
        self.assertEqual(self.sequence(), "ATTCGG")

        del transaction
        gc.collect()

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, self.base_operation.id)

    def test_stale_transaction_drop_cannot_roll_back_a_later_transaction(self):
        stale_transaction = self.repository.transaction()
        stale_transaction.__enter__()
        self.graph.replace("vector:1-3", "TT")
        stale_transaction.__exit__(None, None, None)

        current_transaction = self.repository.transaction()
        current_transaction.__enter__()
        self.graph.replace("vector:4-6", "AA")
        del stale_transaction
        gc.collect()

        self.assertEqual(self.sequence(), "AACCAA")
        current_transaction.__exit__(None, None, None)
        self.assertEqual(self.sequence(), "AACCGG")

    def test_transaction_commit_after_raw_rollback_reports_closed(self):
        head = self.repository.current_branch.head
        operation_count = len(self.repository.get_operations())
        transaction = self.repository.transaction()
        transaction.__enter__()
        self.graph.replace("vector:1-3", "TT")
        self.repository.execute("ROLLBACK")

        with self.assertRaisesRegex(RuntimeError, "closed or stale"):
            transaction.commit("must not commit after raw rollback")

        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(self.repository.current_branch.head, head)
        self.assertEqual(len(self.repository.get_operations()), operation_count)

    def test_stale_context_exit_cannot_clear_a_later_transaction(self):
        stale_transaction = self.repository.transaction()
        with stale_transaction:
            self.graph.replace("vector:1-3", "TT")

        current_transaction = self.repository.transaction()
        stale_transaction.__exit__(None, None, None)
        with current_transaction as transaction:
            self.graph.replace("vector:4-6", "AA")
            committed = transaction.commit("keep later edit")

        self.assertEqual(self.sequence(), "AACCAA")
        self.assertEqual(committed.message, "keep later edit")
        self.assertEqual(
            len(self.repository.get_operations()), self.base_operation_count + 1
        )
