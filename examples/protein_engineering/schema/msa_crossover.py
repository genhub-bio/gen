#!/usr/bin/env python3

import unittest
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import argparse
import os


def gapped_indices(seq):
    """Converts a sequence with gaps to a list of indices, where gaps are represented as None"""
    indices = []
    counter = 1  # 1-based indexing
    for char in seq:
        if char == "-":
            indices.append(None)
        else:
            indices.append(counter)
            counter += 1
    return indices


def translate_coordinates(from_seq, to_seq, coordinates):
    # Strip gaps from the first sequence:
    reference_seq = from_seq.replace("-", "")
    # Sanity check coordinates:
    for coord in coordinates:
        if coord < 1:
            raise ValueError(
                f"Invalid coordinate {coord}: must be positive (one-based indexing)"
            )
        if coord > len(reference_seq):
            raise ValueError(
                f"Invalid coordinate {coord}: does not exist in the reference sequence"
            )

    # Convert both sequences to a list of index coordinates and gaps:
    from_indices = gapped_indices(from_seq)
    to_indices = gapped_indices(to_seq)

    # Translate the given coordinates:
    translated_coordinates = []
    for coord in coordinates:
        alignment_coord = from_indices.index(coord)
        translated_coord = to_indices[alignment_coord]
        translated_coordinates.append(translated_coord)

    return translated_coordinates


def split_at_crossovers(seq, crossovers):
    """Splits a sequence after the given crossovers (one-based indexing)"""

    # Remove any gaps in case the sequence comes from an alignment
    seq = seq.replace("-", "")

    subsequences = []
    start = 0
    for x in crossovers:
        subsequences.append(seq[start:x])
        start = x
    subsequences.append(seq[start:])
    return subsequences


def write_segments_to_disk(sequences, crossovers, output_name):
    reference_id = sequences[0][0]
    reference_seq = sequences[0][1]

    # Set up the output files
    segments_file = f"{output_name}.fa"
    layout_file = f"{output_name}.csv"

    # Clear the output files if they already exist so we can write new data
    for f in [segments_file, layout_file]:
        if os.path.exists(f):
            os.remove(f)

    for id, seq in sequences:
        translated_crossovers = translate_coordinates(reference_seq, seq, crossovers)
        segments = split_at_crossovers(seq, translated_crossovers)
        # Each segment receives a unique ID based on the original sequence ID
        # and the starting coordinate (zero-based indexing to match gen GFAs)
        segment_starts = [0] + [x - 1 for x in translated_crossovers]
        segment_ids = [f"{id}.{i}" for i in segment_starts]

        # Make sure there are no commas in the sequence IDs (this shouldn't be the case)
        if any("," in segment_id for segment_id in segment_ids):
            raise ValueError("Commas are not allowed in sequence IDs")

        # Write each segment to a fasta file
        with open(segments_file, "a") as f:
            for segment_id, segment in zip(segment_ids, segments):
                record = SeqRecord(segment, id=segment_id, description="")
                SeqIO.write(record, f, "fasta")

        # Write the list of segments to a layout csv file
        with open(layout_file, "a") as f:
            f.write(",".join(segment_ids) + "\n")


class TestGappedIndices(unittest.TestCase):
    def test_gapped_indices(self):
        test_seq = "A-BCDE"
        expected_indices = [1, None, 2, 3, 4, 5]
        result = gapped_indices(test_seq)
        self.assertEqual(result, expected_indices)


class TestTranslateCoordinates(unittest.TestCase):
    def test_translate_coordinates(self):
        test_msa = ["A-BCDE", "FGH-IJ"]
        coordinates = [1, 2, 3, 4, 5]
        expected_translation = [1, 3, None, 4, 5]
        result = translate_coordinates(test_msa[0], test_msa[1], coordinates)
        self.assertEqual(result, expected_translation)

    def test_translate_coordinates_reference(self):
        reference = "A-BCDE"
        coordinates = [1, 2, 3, 4, 5]
        expected_translation = coordinates
        result = translate_coordinates(reference, reference, coordinates)
        self.assertEqual(result, expected_translation)


class TestSplitAtCrossover(unittest.TestCase):
    def test_split_at_crossovers(self):
        test_seq = "ABCDEFG"
        test_crossovers = [2, 4]
        expected_subsequences = ["AB", "CD", "EFG"]
        result = split_at_crossovers(test_seq, test_crossovers)
        self.assertEqual(result, expected_subsequences)

    def test_split_at_crossovers_gapped(self):
        test_seq = "ABCDE-FG"
        test_crossovers = [2, 4]
        expected_subsequences = ["AB", "CD", "EFG"]
        result = split_at_crossovers(test_seq, test_crossovers)
        self.assertEqual(result, expected_subsequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create crossover segments from a multiple sequence alignment"
    )
    parser.add_argument(
        "msa", type=str, help="Multiple sequence alignment in fasta format"
    )
    parser.add_argument(
        "crossovers",
        type=int,
        nargs="+",
        help="One or more crossover points to translate",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output name prefix (used for <name>.fa and <name>.csv)",
    )
    args = parser.parse_args()

    sequences = [(r.id, r.seq) for r in SeqIO.parse(open(args.msa), "fasta")]

    write_segments_to_disk(sequences, args.crossovers, args.output)
