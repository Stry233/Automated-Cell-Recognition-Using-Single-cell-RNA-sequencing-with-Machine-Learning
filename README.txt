Files:

- example.py	An example of how to read the data files (requires pandas package)
- all_data.h5	Data file, all of the data (train + test)
- train_data.h5	Data file, training split of the data
- test_data.h5	Data file, testing split of the data

Our samples are single cells.
    Features (variables): the gene expression (in RPKM units) of each of the 20499 genes for a cell
    Label               : the "type" of this single cell (i.e. "neuron", "kidney", etc)


Total Cells: 24244
Train Cells: 21389
Test Cells:   2855


How the data was split:

In order to split, we want to make sure that the test samples come from different studies (read: "publication"/"experiment"/"lab"),
because in practice, a classifier for cell types would be used to predict the cell type of cells from new experiments.

Thus, for each cell-type(label) present in our data, we look at which studies it comes from, and hold some of those out.

NOTE: because of this constraint, not all cell types can be in both train and test (some cell types only come from a single study), thus the test labels are a subset of the train labels.

Below is how I have split it, where "*" in front of an experiment (also called an "accession") indicates that it is held-out for the test set. I've tried to be as close to a 80/20 split per label as possible.


Accessions (experiment) for each cell type, and count of cells in each accession:
CL:0000353 blastoderm cell
        <acsn>: <count>
        *22182 :  10
         45719 :  37
         57249 :  39
CL:0002322 embryonic stem cell
        <acsn>: <count>
         75804 :  2
         57609 :  3
         61346 :  3
         66582 :  4
         66734 :  4
         72852 :  6
         85627 :  9
         72854 :  10
         53386 :  14
         54695 :  16
         61844 :  16
         78140 :  18
         55291 :  28
         72855 :  28
         80280 :  40
         60066 :  55
         47835 :  87
        *29087 :  130
         94579 :  139
        *80168 :  228
         60749 :  296
         81275 :  654
UBERON:0002107 liver
        <acsn>: <count>
        *84498 :  4
         45719 :  7
        *75659 :  16
         96981 :  96
CL:0000057 fibroblast
        <acsn>: <count>
         45719 :  9
         75659 :  207
CL:0000081 blood cell
        <acsn>: <count>
         51372 :  93
UBERON:0001851 cortex
        <acsn>: <count>
         79812 :  2
         56638 :  4
         98969 :  10
        *52564 :  13
        *79374 :  39
        *90860 :  214
         71585 :  1593
UBERON:0000115 lung epithelium
        <acsn>: <count>
        *61300 :  78
         52583 :  142
UBERON:0000922 embryo
        <acsn>: <count>
         66582 :  15
        *53386 :  18
        *69970 :  42
         70713 :  235
CL:0001056 dendritic cell, human
        <acsn>: <count>
         54006 :  24
CL:0000746 cardiac muscle cell
        <acsn>: <count>
         70930 :  6
         56638 :  8
        *92247 :  11
         69926 :  22
UBERON:0001954 Ammon's horn
        <acsn>: <count>
         79812 :  2
        *56638 :  15
         79374 :  49
CL:0002034 long term hematopoietic stem cell
        <acsn>: <count>
         59114 :  798
CL:0002033 short term hematopoietic stem cell
        <acsn>: <count>
         59114 :  696
CL:0000037 hematopoietic stem cell
        <acsn>: <count>
         64002 :  17
        *66954 :  17
         87631 :  91
        *68981 :  145
         59114 :  537
CL:1000497 kidney cell
        <acsn>: <count>
         64959 :  9
         59127 :  84
CL:0008019 mesenchymal cell
        <acsn>: <count>
         61346 :  1
         59129 :  49
UBERON:0000044 dorsal root ganglion
        <acsn>: <count>
         63576 :  5
        *71453 :  123
         59739 :  1006
CL:0002365 medullary thymic epithelial cell
        <acsn>: <count>
         60297 :  160
UBERON:0000473 testis
        <acsn>: <count>
         64960 :  2
         82174 :  423
UBERON:0000992 female gonad
        <acsn>: <count>
         64960 :  141
CL:0002321 embryonic cell
        <acsn>: <count>
        *65924 :  173
         75790 :  514
UBERON:0001003 skin epidermis
        <acsn>: <count>
         90847 :  8
        *67602 :  678
         90848 :  1088
CL:0002319 neural cell
        <acsn>: <count>
         90053 :  16
        *67833 :  81
         76005 :  507
UBERON:0002048 lung
        <acsn>: <count>
        *98048 :  58
         69761 :  66
         78045 :  139
CL:0000137 osteocyte
        <acsn>: <count>
         70240 :  27
         70241 :  28
         90140 :  35
        *78907 :  45
        *70242 :  63
         90797 :  319
UBERON:0001898 hypothalamus
        <acsn>: <count>
        *87544 :  5
        *90809 :  24
         74672 :  1430
UBERON:0001997 olfactory epithelium
        <acsn>: <count>
         75413 :  10
         95601 :  836
CL:0000540 neuron
        <acsn>: <count>
        *75901 :  50
        *79108 :  83
         92707 :  161
         78521 :  288
UBERON:0004129 growth plate cartilage
        <acsn>: <count>
         76157 :  171
UBERON:0001891 midbrain
        <acsn>: <count>
         76381 :  882
UBERON:0002038 substantia nigra
        <acsn>: <count>
         76381 :  224
UBERON:0000007 pituitary gland
        <acsn>: <count>
         77357 :  144
CL:0000763 myeloid cell
        <acsn>: <count>
         77847 :  86
UBERON:0001264 pancreas
        <acsn>: <count>
        *78510 :  162
         87375 :  287
UBERON:0000045 ganglion
        <acsn>: <count>
         78845 :  257
CL:0000127 astrocyte
        <acsn>: <count>
         79374 :  16
CL:0000163 endocrine cell
        <acsn>: <count>
         79457 :  180
CL:0000235 macrophage
        <acsn>: <count>
         94135 :  29
        *79510 :  42
         97941 :  244
UBERON:0000955 brain
        <acsn>: <count>
         79818 :  4
        *79812 :  38
         98969 :  59
UBERON:0000966 retina
        <acsn>: <count>
        *80232 :  250
         81903 :  340
UBERON:0002435 striatum
        <acsn>: <count>
         82187 :  3667
UBERON:0010743 meningeal cluster
        <acsn>: <count>
         83948 :  98
CL:0000169 type B pancreatic cell
        <acsn>: <count>
         87375 :  14
         86479 :  401
CL:0000084 T cell
        <acsn>: <count>
         89405 :  283
UBERON:0001902 epithelium of small intestine
        <acsn>: <count>
         90856 :  400
CL:0000192 smooth muscle cell
        <acsn>: <count>
         97955 :  80

Test cell types: 
         CL:0000353 blastoderm cell
         CL:0002322 embryonic stem cell
         UBERON:0002107 liver
         UBERON:0001851 cortex
         UBERON:0000115 lung epithelium
         UBERON:0000922 embryo
         CL:0000746 cardiac muscle cell
         UBERON:0001954 Ammon's horn
         CL:0000037 hematopoietic stem cell
         UBERON:0000044 dorsal root ganglion
         CL:0002321 embryonic cell
         UBERON:0001003 skin epidermis
         CL:0002319 neural cell
         UBERON:0002048 lung
         CL:0000137 osteocyte
         UBERON:0001898 hypothalamus
         CL:0000540 neuron
         UBERON:0001264 pancreas
         CL:0000235 macrophage
         UBERON:0000955 brain
         UBERON:0000966 retina
Total test cell types:  21
