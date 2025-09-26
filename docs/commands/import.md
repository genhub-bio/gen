# Importing a file

Run `gen import <file type> <file name>` to add a file as a new sequence or set of sequences.  Edits and variants can then be made to the imported sequence(s).

All import commands have an optional `--sample` (`-s`) flag to specify the name
of a new sample to associate the import with, and an optional `--name` (`-n`)
flag to specify which collection to associate the import with.

## FASTA

Imports from a fasta file.  The entire sequence is added to the local gen database.  Example:

`gen import fasta cen-pk113-7d.fa`

`gen import fasta cen-pk113-7d.fa --name reference-yeast-genome --sample original`
Also takes an optional `--shallow` flag.  In the presence of that flag, the
sequence isn't added to the local gen database, but a reference to the file is
saved.  Any operations needing the sequence will read from the file.

`gen import fasta hg39.fa --shallow`

## Genbank

Imports from a Genbank file.  All sequence data is added to the local gen
database.  There are currently no options specific to this format.

`gen import genbank edits.gb`

## GFA

Imports from a GFA file.  All sequence data is added to the local gen database.
There are currently no options specific to this format.

`gen import gfa pangenome.gfa`

## Combinatorial library

Imports from a fasta file with a list of parts, and a CSV file specifying a
combinatorial design from those parts.  You must also specify a region name that
will be used to reference the sequence graph in gen.  All sequence data is added
to the local gen database.  There are currently no options specific to this
format.

`gen import library library-design-123 parts.fa layout.csv`
