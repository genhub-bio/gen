"""Sample copying and operation rollback without sequence mutation."""

from test_api import FIXTURES, RepositoryTestCase


class SampleCopyTests(RepositoryTestCase):
    def setUp(self):
        super().setUp()
        self.sample = self.repository.import_fasta(str(FIXTURES / "simple.fa"))

    def test_copy_preserves_sequence_and_records_one_operation(self):
        before = len(self.repository.get_operations())
        child = self.sample.copy("child", message="Copy for review")
        self.assertEqual(child.sample_name, "child")
        self.assertEqual(child.collection_name, self.sample.collection_name)
        self.assertEqual(
            [graph.name for graph in child], [graph.name for graph in self.sample]
        )
        self.assertEqual(
            child[0].region("m123:0-34").sequence,
            self.sample[0].region("m123:0-34").sequence,
        )
        self.assertEqual(len(self.repository.get_operations()), before + 1)

    def test_copy_rejects_existing_or_invalid_names_without_an_operation(self):
        self.sample.copy("child")
        before = len(self.repository.get_operations())
        for name in ("child", "", self.sample.sample_name):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.sample.copy(name)
        self.assertEqual(len(self.repository.get_operations()), before)

    def test_operation_recording_failure_rolls_back_sample_copy(self):
        before = len(self.repository.get_operations())
        self.repository.execute(
            "CREATE TEMP TRIGGER reject_copy BEFORE INSERT ON gen_operation_log "
            "BEGIN SELECT RAISE(ABORT, 'recording failed'); END"
        )
        with self.assertRaisesRegex(RuntimeError, "recording failed"):
            self.sample.copy("child")
        self.assertEqual(len(self.repository.get_operations()), before)
        self.assertNotIn(
            "child", [sample.sample_name for sample in self.repository.samples]
        )
        self.repository.execute("DROP TRIGGER reject_copy")
        self.assertEqual(self.sample.copy("child").sample_name, "child")
