## Sequence graphs vs. segment graphs

Gen represents sequences as a sequence graph. Nodes hold sequence fragments, edges connect them, and any linear sequence is reconstructed by walking a defined path. New variants extend the graph without splitting existing nodes, so node IDs remain stable across updates.

![Figure 1](figures/figure_1.svg)

**_Figure 1_**: _Sequence graph representation of a variant where two nucleotides AT are replaced by TG; the modified sequence (shown in bold) is stored as a path over a list of edges that address specific coordinates._

This differs from the segment graph model used by tools like vg and Bandage, where the reference sequence is split into pieces to accommodate each variant. Gen converts between the two formats on GFA export.

![Figure 2](figures/figure_2.svg)

**_Figure 2_**: _Segment graph model corresponding to the variant in Figure 1. The original sequence is split into 3 parts; the modified path is defined by a list of nodes that refer to these segments._

## Coordinates

Indexing into a graph is one of the most challenging parts of working with graph genomes. As an example, take the following
reference genome with variants found in a sample:

```mermaid
flowchart
    subgraph sample-1
    ATC --> GAA
    GAA --> TTGCATG
    ATC ---|deletion|TTGCATG
    TTGCATG ---|insertion-1|AAA --> ACATACA
    TTGCATG ---|insertion-2| CAAAGA --> ACATACA
    end
    subgraph reference-genome
    ATCGAATTGCATGACATACA
    end
```

Now, suppose we want to perform engineering on sample-1 and wish to carry out an insertion at the `CATA` position. How
can this position be addressed using the linear coordinate space? Because each chromosome's path is a different length due
to variants, there is no common reference frame for this position. To try and address this issue, the following
conventions are used.

* If the region is not altered, the base genome can be referenced explicitly. For example, within sample-1, `CATA` can 
be referenced as position 15-19 as that region is not impacted by any variants.
* If the organism has a single copy of the genetic material, sample-1 can be accessed assuming all variants have been
incorporated. Thus, if the reference genome has an `A` at position 100 and there is a 1 basepair deletion at position 90,
in sample-1, that position can be referenced 
* To refer to positions within alterations where no unambiguous coordinate system exists, `accessions` can be utilized 
to provide named regions for modifications. Coordinates within accessions are relative to the accession itself. 
Therefore, if we named the `CAAGA` insertion as `insertion-2`, the name `insertion-2:3-5` would refer to positions `AG`.

Mutability
----------

Changes made in this manner will mutate the graph for a given sample, thus operations will not be communitive. For 
instance, if we have sequence `ATCG` and insert `AA` at position 2 to make it `ATAACG`, a subsequent insertion at
position 3 will not reference the `G` in `ATCG`, but the `A` in `ATA`. For a given set of changes (such as a single
vcf file), the coordinate scheme will be consistent. For instance, if the previous 2 changes were in the same update
batch, 3 will refer to `G` in `ATCG`. But if the changes were split into 2 operations, the second case would apply.
