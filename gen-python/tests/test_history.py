"""Repository branch and operation changes preserve the selected graph state."""

import gen

from test_api import RepositoryTestCase


class RepositoryHistoryTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.repository.import_sequence("AACCGG", name="vector", sample="parent")
        self.base_operation = self.repository.get_operations()[0]

    def sequence(self, repository=None):
        repository = repository or self.repository
        [graph] = repository.get_sequence_graphs(name="vector", sample="parent")
        [sequence] = graph.all_sequences()
        return str(sequence)

    def edit_vector(self, repository=None):
        repository = repository or self.repository
        [graph] = repository.get_sequence_graphs(name="vector", sample="parent")
        graph.replace("vector:1-3", "TT", message="replace motif")
        return repository.get_operations()[0]

    def test_checkout_starts_branches_at_historical_operations_and_persists(self):
        self.edit_vector()
        references = (
            self.base_operation,
            self.base_operation.id,
            str(self.base_operation.id),
        )

        for index, reference in enumerate(references):
            with self.subTest(reference_type=type(reference).__name__):
                branch = self.repository.checkout(
                    f"checkpoint-{index}", create=True, start=reference
                )
                self.assertEqual(branch.head, self.base_operation.id)
                self.assertEqual(self.sequence(), "AACCGG")
                self.repository.checkout("main")

        selected = self.repository.checkout("checkpoint-0")
        reopened = gen.Repository(str(self.root / "repository"))
        self.assertEqual(reopened.current_branch.name, selected.name)
        self.assertEqual(self.sequence(reopened), "AACCGG")
        reopened.checkout("main")
        self.assertEqual(self.sequence(reopened), "ATTCGG")

    def test_checkout_rejects_invalid_start_without_creating_a_branch(self):
        self.edit_vector()
        branch_names = {branch.name for branch in self.repository.get_branches()}
        missing_operation = "1" * 40

        with self.assertRaisesRegex(ValueError, "start can only be used"):
            self.repository.checkout("invalid", start=self.base_operation)
        with self.assertRaisesRegex(ValueError, "operation .* not found"):
            self.repository.checkout("invalid", create=True, start=missing_operation)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.repository.checkout("main", create=True, exist_ok=True, start="main")

        self.assertEqual(
            {branch.name for branch in self.repository.get_branches()}, branch_names
        )
        self.assertEqual(self.repository.current_branch.name, "main")
        self.assertEqual(self.sequence(), "ATTCGG")

    def test_merge_adopts_sequence_data_from_another_branch(self):
        self.repository.checkout("feature", create=True)
        self.repository.import_sequence("TTGCAA", name="feature-only", sample="feature")
        self.repository.checkout("main")
        self.repository.import_sequence("GGGG", name="main-only", sample="main-sample")

        merged = self.repository.merge("feature")
        reopened = gen.Repository(str(self.root / "repository"))
        self.assertEqual(reopened.current_branch.head, merged.id)
        [graph] = reopened.get_sequence_graphs(name="feature-only", sample="feature")
        [sequence] = graph.all_sequences()
        self.assertEqual(str(sequence), "TTGCAA")
        [graph] = reopened.get_sequence_graphs(name="main-only", sample="main-sample")
        [sequence] = graph.all_sequences()
        self.assertEqual(str(sequence), "GGGG")

    def test_apply_accepts_operation_hash_and_string_references(self):
        references = ("operation", "hash", "string")
        for index, reference_type in enumerate(references):
            with self.subTest(reference_type=reference_type):
                repository = gen.Repository(str(self.root / f"apply-{index}"))
                repository.import_sequence("AACCGG", name="vector", sample="parent")
                base_operation = repository.get_operations()[0]
                repository.checkout("feature", create=True)
                [graph] = repository.get_sequence_graphs(name="vector", sample="parent")
                graph.replace("vector:1-3", "TT", message="replace motif")
                operation = repository.get_operations()[0]
                repository.checkout("main")

                reference = {
                    "operation": operation,
                    "hash": operation.id,
                    "string": str(operation.id),
                }[reference_type]
                applied = repository.apply(reference)

                self.assertNotEqual(applied.id, base_operation.id)
                [current] = repository.get_sequence_graphs(
                    name="vector", sample="parent"
                )
                [sequence] = current.all_sequences()
                self.assertEqual(str(sequence), "ATTCGG")

    def test_reset_accepts_operation_hash_and_string_references(self):
        references = ("operation", "hash", "string")
        for index, reference_type in enumerate(references):
            with self.subTest(reference_type=reference_type):
                repository = gen.Repository(str(self.root / f"reset-{index}"))
                repository.import_sequence("AACCGG", name="vector", sample="parent")
                base_operation = repository.get_operations()[0]
                [graph] = repository.get_sequence_graphs(name="vector", sample="parent")
                graph.replace("vector:1-3", "TT", message="replace motif")
                reference = {
                    "operation": base_operation,
                    "hash": base_operation.id,
                    "string": str(base_operation.id),
                }[reference_type]

                reset = repository.reset(reference)

                self.assertEqual(reset.id, base_operation.id)
                self.assertTrue(reset.is_head)
                [current] = repository.get_sequence_graphs(
                    name="vector", sample="parent"
                )
                [sequence] = current.all_sequences()
                self.assertEqual(str(sequence), "AACCGG")

                reopened = gen.Repository(str(self.root / f"reset-{index}"))
                self.assertEqual(reopened.current_branch.head, base_operation.id)
                [current] = reopened.get_sequence_graphs(name="vector", sample="parent")
                [sequence] = current.all_sequences()
                self.assertEqual(str(sequence), "AACCGG")

    def test_force_reset_discards_dirty_state_only_for_a_valid_operation(self):
        self.edit_vector()
        self.repository.execute(
            "INSERT INTO collections (name) VALUES ('uncommitted collection')"
        )
        head_before = self.repository.current_branch.head

        with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
            self.repository.checkout("main")
        with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
            self.repository.checkout("orphan", create=True)
        with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
            self.repository.merge("main")
        with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
            self.repository.apply(self.base_operation)

        self.assertEqual(
            {branch.name for branch in self.repository.get_branches()}, {"main"}
        )
        self.assertEqual(self.repository.current_branch.name, "main")
        self.assertEqual(self.repository.current_branch.head, head_before)
        self.assertTrue(self.repository.current_branch.dirty)
        self.assertEqual(
            self.repository.query(
                "SELECT name FROM collections WHERE name = 'uncommitted collection'"
            ),
            [["uncommitted collection"]],
        )

        with self.assertRaisesRegex(RuntimeError, "uncommitted changes"):
            self.repository.reset(self.base_operation)
        self.assertTrue(self.repository.current_branch.dirty)

        with self.assertRaisesRegex(ValueError, "operation .* not found"):
            self.repository.reset("1" * 40, force=True)
        self.assertTrue(self.repository.current_branch.dirty)
        self.assertEqual(self.repository.current_branch.head, head_before)

        reset = self.repository.reset(self.base_operation, force=True)

        self.assertEqual(reset.id, self.base_operation.id)
        self.assertFalse(self.repository.current_branch.dirty)
        self.assertEqual(self.sequence(), "AACCGG")
        self.assertEqual(
            self.repository.query(
                "SELECT name FROM collections WHERE name = 'uncommitted collection'"
            ),
            [],
        )
