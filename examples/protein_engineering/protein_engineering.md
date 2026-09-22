# Tracking protein variant libraries
## Site-directed Homologous Recombination

This example recreates a protein engineering library made through site-directed, homologous recombination guided by
structure-based computation (SCHEMA) ([Otey 2006](https://doi.org/10.1371/journal.pbio.0040112)). Starting from three existing
cytochrome P450 proteins, approximately 3,000 artifical (chimeric) proteins were constructed and tested. The authors
describe it as follows: 

> _"We generated an artificial family of cytochromes P450 by recombining fragments of the genes encoding the
> heme-binding domains of three bacterial P450s, CYP102A1 (also known as P450BM3), CYP102A2, and CYP102A3 (abbreviated
> A1, A2, and A3), which share ̃65% amino acid identity [...] The final design has crossovers located after residues
> Glu64, Ile122, Tyr166, Val216, Thr268, Ala328, and Gln404, based on the numbering of the A1 sequence"_

First we download the sequences of the parent proteins and save them as one fasta file:

```console
wget https://rest.uniprot.org/uniprotkb/P14779.fasta https://rest.uniprot.org/uniprotkb/O08394.fasta https://rest.uniprot.org/uniprotkb/O08336.fasta
cat O08336.fasta O08394.fasta P14779.fasta > parents.fa
```

Next we create multiple sequence alignment using the Muscle application, which you can run for example through Docker:

```console
 docker run --rm --volume $PWD:/data --workdir /data pegi3s/muscle -in parents.fa -out parents_aligned.fa
```

With this alignment, we can then translate the crossover points from the A1 reference frame to all other proteins. The
msa_crossover.py Python script performs those calculations, creates the protein segments, and saves them to disk in a
format readable by the gen update command. In the future this functionality may be incorporated in the gen client
directly.

```console
python msa_crossover.py parents_aligned.fa 64 122 166 216 268 328 404
```

The default output of this script is a directory called 'output' that contains the files 'layout.csv' and 'segments.fa'.
We now set up our gen repository, create a new branch and switch into it. Then we import one of the parents to have a 
starting point, and perform an update operation. The name of the target path is derived from the fasta file we just
imported, and because it contains | (pipe) symbols we must wrap it in quotes to not confuse the shell. As start and end 
coordinate we choose 0 and 657 because want to replace the entire 1049 residue protein with the combinatorial library.
The parts and library files were obtained by running the msa_crossover.py script as shown above, and the resulting
modifications will be stored as a new virtual sample called schema_library. Lastly, we export that sample to a GFA file.

```console
gen init
gen branch --create ex1
gen branch --checkout ex1
gen import fasta P14779.fasta
gen update --path-name "sp|P14779|CPXB_PRIM2" --start 0 --end 1049 --parts output/segments.fa --library output/layout.csv --new-sample schema_library
gen export --sample schema_library --gfa P450_chimera.gfa
```

## Site Saturation Mutagenesis
In the next example we will demonstrate the representation of a Site Saturation Mutagenesis library in gen by recreating
an experiment described in ([Wu 2016](https://doi.org/10.7554/eLife.16965)):

> _"In this study, we investigated the fitness landscape of all variants (20^4 = 160,000) at four amino acid sites (V39,
> D40, G41 and V54) in an epistatic region of protein G domain B1 (GB1, 56 amino acids in total)"_

We start by switching back to the main branch and creating a new experimental branch. Then we download the reference 
sequence of the B1 domain of immunoglobulin G-binding protein G found in _Streptococcus_, and import it into our new 
branch. We conclude by listing the branches of our repository, which shows that we have so far run 3 operations in the 
repository overall, with operation 3 taking place in branch `ex2`: 

```console
gen branch --checkout main
gen branch --create ex2
gen branch --checkout ex2
wget https://www.rcsb.org/fasta/entry/1PGA -O GB1.fa
gen import --fasta GB1.fa
gen branch --list

   Name                             Operation           
   ex1                              2                   
>  ex2                              3                   
   main                             -1          
```

To mutagenize the 2 sites (V54 and V39-D40-G41) we will perform two update operations using a [parts file](saturation_parts.fa) that defines a sequence of length 1 for every amino acid, and two layout files for [one](saturation_single.csv) and [three](saturation_triple.csv) consecutive residues. Please note that gen uses 0-based indexing, so residue 39 is passed.
as `--start 38`. 


```console
gen update --path-name "1PGA_1|Chain" --start 38 --end 41 --parts saturation.fa --library saturation_triple.csv --new-sample gb1_mut1
```





# Bibliography

Otey, C. R., Landwehr, M., Endelman, J. B., Hiraga, K., Bloom, J. D., & Arnold, F. H. (2006). Structure-guided
recombination creates an artificial family of cytochromes P450. PLoS biology, 4(5), e112.
https://doi.org/10.1371/journal.pbio.0040112 

Wu, N. C., Dai, L., Olson, C. A., Lloyd-Smith, J. O., & Sun, R. (2016). Adaptation in protein fitness landscapes is
facilitated by indirect paths. Elife, 5, e16965.  https://doi.org/10.7554/eLife.16965


    
