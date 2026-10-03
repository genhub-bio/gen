"""Re-running a notebook cell: `exist_ok=True` reuses what the first run created."""

from test_api import RepositoryTestCase


class ExistOkTests(RepositoryTestCase):
    def test_import_sequence_exist_ok_returns_the_identical_graph(self):
        first = self.repository.import_sequence("AAAACCCC", name="v", sample="p")
        operations = len(self.repository.get_operations())
        again = self.repository.import_sequence(
            "AAAACCCC", name="v", sample="p", exist_ok=True
        )
        self.assertEqual(again.id, first.id)
        self.assertEqual(len(self.repository.get_operations()), operations)

    def test_import_sequence_exist_ok_rejects_a_different_sequence(self):
        self.repository.import_sequence("AAAACCCC", name="v", sample="p")
        with self.assertRaisesRegex(RuntimeError, "different sequence"):
            self.repository.import_sequence("GGGG", name="v", sample="p", exist_ok=True)

    def test_import_sequence_without_exist_ok_explains_the_options(self):
        self.repository.import_sequence("AAAACCCC", name="v", sample="p")
        with self.assertRaisesRegex(RuntimeError, "exist_ok=True"):
            self.repository.import_sequence("AAAACCCC", name="v", sample="p")

    def test_copy_exist_ok_returns_the_existing_sample_unchanged(self):
        graph = self.repository.import_sequence("AAAACCCC", name="v", sample="p")
        source = next(s for s in self.repository.samples if s.name == "p")
        design = source.copy("design")
        design[0].replace("v:0-4", "GGGG")
        again = source.copy("design", exist_ok=True)
        self.assertEqual(again[0].region("v:0-4").sequence, "GGGG")
        self.assertEqual(graph.region("v:0-4").sequence, "AAAA")
        with self.assertRaisesRegex(ValueError, "exist_ok=True"):
            source.copy("design")

    def test_checkout_create_exist_ok_switches_to_the_existing_branch(self):
        self.repository.checkout("work", create=True)
        self.repository.checkout("main")
        branch = self.repository.checkout("work", create=True, exist_ok=True)
        self.assertEqual(branch.name, "work")
        self.assertTrue(branch.is_current)
        with self.assertRaisesRegex(RuntimeError, "exist_ok=True"):
            self.repository.checkout("work", create=True)
