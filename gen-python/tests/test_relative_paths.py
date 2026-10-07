"""File arguments given as relative paths mean the process's current directory."""

import os
import shutil

from test_api import FIXTURES, RepositoryTestCase


class RelativePathTests(RepositoryTestCase):
    """The repository lives in a subdirectory, so the workspace is never the cwd."""

    def setUp(self):
        super().setUp()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root)

    def stage(self, *names):
        for name in names:
            shutil.copy(FIXTURES / name, self.root / name)

    def import_parent(self):
        self.repository.import_reference_fasta(str(FIXTURES / "simple.fa"), "ref")
        return self.repository.import_fasta(
            str(FIXTURES / "simple.fa"), sample="parent"
        )

    def test_import_fasta_reads_a_file_in_the_current_directory(self):
        self.stage("simple.fa")
        sample = self.repository.import_fasta("simple.fa", sample="relative")
        self.assertEqual(sample.name, "relative")

    def test_import_reference_fasta_reads_a_file_in_the_current_directory(self):
        self.stage("simple.fa")
        sample = self.repository.import_reference_fasta("simple.fa", "relative_ref")
        self.assertEqual(sample.name, "relative_ref")

    def test_import_genbank_reads_a_file_in_the_current_directory(self):
        self.stage("puc19.gb")
        sample = self.repository.import_genbank("puc19.gb", sample="plasmid")
        self.assertEqual(sample.name, "plasmid")

    def test_import_gfa_reads_a_file_in_the_current_directory(self):
        self.stage("simple.gfa")
        graph = self.repository.import_gfa("simple.gfa", sample="graph")
        self.assertEqual(graph.sample.name, "graph")

    def test_import_annotations_reads_a_file_in_the_current_directory(self):
        self.import_parent()
        self.stage("simple.gff")
        self.repository.import_annotations("simple.gff")
        names = [asset.name for asset in self.repository.get_assets()]
        self.assertIn("simple.gff", names)

    def test_import_library_files_reads_files_in_the_current_directory(self):
        self.stage("affix_parts.fa", "affix_layout.csv")
        graph = self.repository.import_library_files(
            "library", "affix_parts.fa", "affix_layout.csv", sample="designs"
        )
        self.assertEqual(graph.sample.name, "designs")

    def test_add_file_reads_a_file_in_the_current_directory(self):
        self.stage("simple.gff")
        asset = self.repository.add_file("simple.gff")
        self.assertEqual(asset.name, "simple.gff")

    def test_update_with_fasta_reads_a_file_in_the_current_directory(self):
        self.import_parent()
        self.stage("parts.fa")
        sample = self.repository.update_with_fasta(
            "parts.fa", sample="parent", new_sample="edited", region_name="m123:2-4"
        )
        self.assertEqual(sample.name, "edited")

    def test_update_with_gfa_reads_a_file_in_the_current_directory(self):
        self.import_parent()
        self.stage("simple.gfa")
        sample = self.repository.update_with_gfa(
            "simple.gfa", sample="parent", new_sample="edited"
        )
        self.assertEqual(sample.name, "edited")

    def test_update_with_genbank_reads_a_file_in_the_current_directory(self):
        self.import_parent()
        self.stage("puc19.gb")
        sample = self.repository.update_with_genbank(
            "puc19.gb", sample="parent", create_missing=True
        )
        self.assertEqual(sample.name, "parent")

    def test_update_with_vcf_reads_a_file_in_the_current_directory(self):
        self.import_parent()
        self.stage("simple.vcf")
        samples = self.repository.update_with_vcf("simple.vcf", reference="ref")
        self.assertEqual(
            sorted(sample.name for sample in samples), ["G1", "foo", "unknown"]
        )

    def test_update_with_vcf_raises_for_a_missing_file(self):
        self.import_parent()
        with self.assertRaises(FileNotFoundError):
            self.repository.update_with_vcf("absent.vcf", reference="ref")

    def test_update_with_gaf_reads_files_in_the_current_directory(self):
        self.import_parent()
        self.stage("chr22_het.gaf", "chr22_insert.csv")
        with self.assertRaises(RuntimeError) as raised:
            self.repository.update_with_gaf(
                "chr22_het.gaf", "chr22_insert.csv", sample="parent"
            )
        self.assertNotIn("No such file", str(raised.exception))

    def test_update_with_library_files_reads_files_in_the_current_directory(self):
        self.import_parent()
        self.stage("affix_parts.fa", "affix_layout.csv")
        sample = self.repository.update_with_library_files(
            "parent", "designs", "m123:2-4", "affix_layout.csv", "affix_parts.fa"
        )
        self.assertEqual(sample.name, "designs")

    def test_saving_and_exporting_write_to_the_current_directory(self):
        graph = self.import_parent()[0]
        graph.export_fasta("out.fa")
        graph.export_genbank("out.gb")
        graph.export_gfa("out.gfa")
        asset = self.repository.add_file(str(FIXTURES / "simple.gff"))
        asset.save_as("saved.gff")
        for name in ("out.fa", "out.gb", "out.gfa", "saved.gff"):
            with self.subTest(name=name):
                self.assertTrue((self.root / name).is_file())
