"""Reference aliases let files that use another naming scheme match a graph."""

from test_api import FIXTURES, RepositoryTestCase


class ReferenceAliasTests(RepositoryTestCase):
    def aliased_vcf(self):
        """A copy of simple.vcf whose contig is called chr7 instead of m123."""
        vcf = self.root / "chr7.vcf"
        vcf.write_text((FIXTURES / "simple.vcf").read_text().replace("m123", "chr7"))
        return str(vcf)

    def test_vcf_with_an_unknown_contig_name_fails_without_an_alias(self):
        self.repository.import_reference_fasta(str(FIXTURES / "simple.fa"), "ref")
        with self.assertRaises(RuntimeError):
            self.repository.update_with_vcf(self.aliased_vcf(), reference="ref")

    def test_alias_lets_a_vcf_use_another_name_for_the_graph(self):
        self.repository.import_reference_fasta(str(FIXTURES / "simple.fa"), "ref")
        self.repository.add_reference_alias("contig 7", genbank_id="m123", chromosome=7)
        samples = self.repository.update_with_vcf(self.aliased_vcf(), reference="ref")
        self.assertEqual(
            sorted(sample.name for sample in samples), ["G1", "foo", "unknown"]
        )

    def test_adding_an_alias_is_recorded_as_one_operation(self):
        operations_before = len(self.repository.get_operations())
        self.repository.add_reference_alias("contig 7", custom_id="seven")
        operations = self.repository.get_operations()
        self.assertEqual(len(operations), operations_before + 1)
        self.assertEqual(operations[0].message, "add aliases for reference 'contig 7'")

    def test_alias_without_an_identifier_is_rejected(self):
        with self.assertRaises(ValueError):
            self.repository.add_reference_alias("contig 7")
