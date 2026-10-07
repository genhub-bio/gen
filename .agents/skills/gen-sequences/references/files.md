# Files: import, update from files, export

Signatures were checked against `gen-python/src/python_api/` and the stub
`gen-python/python/gen/gen/__init__.pyi`, which has every signature and docstring. Check
`help()` in the actual interpreter when using a different release. File APIs take filename
strings, not `Path` objects. Use absolute paths: some methods
(`import_fasta`, `update_with_fasta`, `add_file`) resolve a relative filename against the
workspace directory, not the current directory, and others (`import_genbank`) use the current
directory. `import_fasta("plasmid.fa")` fails with "No such file" when the file sits next to
your script.

Contents: choosing an import; imports and return types; Biopython records; annotation files;
updating a sample from a file; keeping other files; exports; mistakes to avoid.

## Choosing an import

| You have | Call | Returns |
|---|---|---|
| a string, `Seq` or `SeqRecord` in memory | `repo.import_sequence()` | `SequenceGraph` |
| a FASTA file (one or many records) | `repo.import_fasta()` | `Sample`, one graph per record |
| a FASTA that other samples are derived against (VCF reference) | `repo.import_reference_fasta()` | `Sample` |
| a GenBank file with features | `repo.import_genbank()` | `Sample` |
| a GFA file with nodes and edges | `repo.import_gfa()` | `SequenceGraph` |
| named parts plus a library design | `repo.import_library()` / `import_library_files()` | `SequenceGraph` (see `design.md`) |
| a GFF/BED/GenBank annotation file for existing graphs | `repo.import_annotations()` | operation `HashId` |
| variants in a VCF | `repo.update_with_vcf()` | `list[Sample]` |

## Imports and return types

```python
repo.import_fasta(filename, sample=None, collection=None)
repo.import_reference_fasta(filename, reference, collection=None)
repo.import_genbank(filename, sample=None, collection=None)
repo.import_gfa(filename, sample=None, collection=None)
repo.import_sequence(sequence, name=None, sample=None, circular=False, collection=None, *, exist_ok=False)
repo.import_reference_sequence(sequence, reference, name=None, circular=False, collection=None, *, exist_ok=False)
repo.import_library(library_name, parts_list, sample=None, collection=None)
repo.import_library_files(library_name, parts, library, sample=None, collection=None)
```

| Import | Return |
|---|---|
| FASTA, reference FASTA, GenBank | `Sample` |
| GFA, library, library files | `SequenceGraph` |
| sequence, reference sequence | `SequenceGraph` |

There is no Python `shallow=` argument on FASTA imports. GenBank imports load
features automatically, accessible through `graph.annotations`. `import_fasta()` fails if the
same contents were already imported, so a re-run in a notebook needs a guard or a fresh
sample name. `import_sequence()` and `import_reference_sequence()` take `exist_ok=True` to
return the earlier graph instead.

`import_sequence()` accepts a string, Biopython `Seq`, or `SeqRecord`. Strings and
`Seq` values need `name`; a `SeqRecord` supplies its id unless overridden. `circular=True`
stores a circular graph (plasmids). It imports sequence, not the record's feature annotations.
Its `sample` argument (and `import_reference_sequence()`'s `reference`) accepts a name or a
`Sample`; a sample object supplies its collection unless explicitly overridden. Do not
assume file imports or updates also accept sample objects: their sample arguments are strings.
Calling `import_sequence()` repeatedly with the same `sample` builds that sample up graph by
graph; duplicate graph names fail. Each call records an operation, so use FASTA for large
file imports. The default sample name is `"reference"`.

```python
graph = repo.import_sequence("AAAACCCCGGGGTTTT", name="vector", sample="parent")
sample = repo.import_fasta("parts.fa", sample="library_parts")
for graph in sample:                    # one graph per FASTA record
    print(graph.name, len(graph.region(f"{graph.name}:0-4")))
```

## Biopython records (import only)

Biopython is for getting records into Gen, not for editing them afterwards. If a
Biopython `Seq` or `SeqRecord` already exists in memory, hand it to Gen as is:

```python
# for record in SeqIO.parse("input.fa", "fasta"):
#     repo.import_sequence(record, sample="records")
```

For a plain FASTA file prefer `repo.import_fasta()`, which makes one call instead of one
operation per record. After import, make every change with the edit methods in `design.md`.

## Annotation files

```python
op = repo.import_annotations(filename, format=None, index=None, name=None, message=None)
graph.annotations        # database and attached-file annotations together
```

`import_annotations()` records an annotation file as an asset and returns the `HashId` of the
operation that recorded it. Format is inferred from the filename unless supplied (`"gff3"`,
`"bed"` or `"genbank"`); a neighboring tabix index is discovered unless an `index` path is
given. `name` controls the display/track name. File reference names must match the graph
context (the sequence names in the file must match graph or path names) to appear in
`.annotations`; if nothing shows up, compare the file's reference names with `graph.name`.
Annotations you create from Python are in `design.md`.

## Updating a sample from a file

```python
repo.update_with_fasta(filename, sample, new_sample, region_name, collection=None)
repo.update_with_genbank(filename, sample, create_missing=False, collection=None)
repo.update_with_gfa(filename, sample, new_sample, collection=None)
repo.update_with_vcf(filename, reference=None, genotype=None, sample=None,
                     in_place=False, collection=None)
repo.update_with_gaf(filename, csv, sample, parent_sample=None, collection=None)
repo.update_with_library(sample, new_sample_name, path_name, parts_list, collection=None)
repo.update_with_library_files(sample, new_sample, path_name, library, parts, collection=None)
```

VCF returns `list[Sample]`; GAF and other updates return a `Sample`. FASTA, GFA, and library
updates create the named output sample. `update_with_fasta()` replaces the region
`region_name` of `sample` with the sequence in the FASTA file. To replace a region with a
literal sequence, `sample.copy()` it and call `graph.replace()` on the copy (`design.md`).
GenBank updates the specified sample; `create_missing=True` allows graphs not yet in the
sample. VCF exposes `in_place`; do not assume every update creates a new sample.

`update_with_vcf()` applies variants to the `reference` sample (a name or list of names),
creating one `Sample` per VCF sample column, or only the column named by `sample`. With
`in_place=True` the reference itself is edited. The reference sample must already exist, for example from
`import_reference_fasta()`.

When a VCF, GFF or BED names a contig differently from the graph (`chr7` in the file, `NC_000007.14`
in the repository), declare the names once instead of editing the file:

```python
repo.add_reference_alias("contig 7", genbank_id="NC_000007.14", chromosome=7)
```

The first argument is only a label. One identifier must equal the graph's own name; the file's
name then matches through the others (`chromosome=7` also matches `chr7`, `Chromosome7` and so
on; `refseq_accession_id` and `genbank_id` also match without their version suffix). Give at
least one identifier. It is recorded as an operation. Without it, a VCF with an unknown contig
fails with "Region not found".

```python
edited = repo.update_with_fasta(
    "parts.fa", sample="parent", new_sample="file-style-design",
    region_name="vector:4-8",
)
# VCF reference names the parent/reference; sample selects the VCF sample column.
# repo.import_reference_fasta("ref.fa", "ref")
# variants = repo.update_with_vcf("variants.vcf", reference="ref", sample="sequenced")
# variant_graph = variants[0][0]
```

## Keeping other files

```python
asset = repo.add_file("protocol.md", message=None)   # stored, not imported as sequence
repo.get_assets()                                    # every asset on the branch
asset.id; asset.name; asset.path
asset.save_as("copy.md", overwrite=False)
```

Assets are files kept in the repository: imported files, annotation files, and anything added
with `repo.add_file()`, which stores a file (a README, a protocol, a plate map) without
importing it as sequence and returns its `Asset`. `Asset` has `id` (`HashId`), `name`
(original file name), `path` (the stored copy under `.gen`, which has a hashed filename;
raises `FileNotFoundError` if not downloaded yet) and `save_as(destination, overwrite=False)`,
which copies it to a path of your choice (or into a directory under `name`). Pull transfers
assets; see `version-control.md`.

## Exports

```python
graph.export_fasta(filename, all_sequences=False)
graph.export_genbank(filename)
graph.export_gfa(filename, node_max=None)
```

All return `None`. Graph-level exports cover the **entire sample**, not just that graph.
FASTA defaults to current named paths; `all_sequences=True` enumerates alternatives
with `"{graph_name}.{index}"` headers (1-based). Use it to inspect stacked edits or
libraries, mindful of the number of combinations. GenBank preserves annotations;
GFA preserves graph structure and is the choice for handing the whole graph to another tool.
For a single region as text, read `graph.region("name:0-100").sequence` instead of exporting.

| Goal | Use |
|---|---|
| order a synthesized construct or send to a collaborator | `export_fasta` (current paths) or `export_genbank` (with features) |
| list every library design | `export_fasta(all_sequences=True)` or `[str(s) for s in graph.all_sequences()]` |
| preserve the graph itself | `export_gfa` |
| one region as a string | `graph.region(...).sequence` |

## Mistakes to avoid

- Reading a FASTA with Biopython, slicing strings, and writing a new FASTA to "edit" it. Import
  it, `sample.copy()`, edit with `replace`/`insert`/`delete`, and export at the end.
- Exporting default FASTA after `stack=True` edits and concluding the edit is missing. Default
  FASTA shows the original path; use `all_sequences=True`.
- Exporting a graph and expecting only that graph. The whole sample is written.
- Importing the same FASTA twice. It fails; reuse the first `Sample`.
- Passing a `Sample` object to a file import or update. Those take sample names as strings.
