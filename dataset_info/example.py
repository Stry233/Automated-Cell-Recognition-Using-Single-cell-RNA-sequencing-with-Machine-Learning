import pandas as pd

# open up a datastore
store = pd.HDFStore('all_data.h5')

# An HDF5 file is a key-value store, and in our case, each value is a Pandas
# object (either a 2D DataFrame, or a 1D Series)

# Keys:
# /rpkm           This contains the data. Each row is a single cell. Each column is the expression of a gene in RPKM units.
#                 The "index" (the row names, primary key of the table) is a unique identifier for each cell. The column names are Entrez IDs,
#                 which are unique numerical identifiers for genes.

# /labels         Vector, same length as number of rows in 'rpkm', contains the correct label (cell type) for each cell.

# /accessions     This is a vector of the same length as number of rows in 'rpkm' and contains the experimentID (accession) for
#                 each cell. This is actually embedded in the unique identifier for the cell, but is included as a separate vector
#                 for convenience.

# /gene_symbols   Might not be useful. Vector, same length as number of columns in 'rpkm', contains the names (strings) for each gene. Just
#                 a different way of identifying the genes.

# /true_ids       Not relevant for you, ignore.
print(store)
print()
print()

# Get the feature matrix (samples and their features)
feature_matrix_dataframe = store['rpkm']
print(type(feature_matrix_dataframe))
print(feature_matrix_dataframe.info())
print()
print()

# Get the labels corresponding to each of the samples in the feature matrix
labels_series = store['labels']
print(type(labels_series))
print(labels_series.shape)
print()
print()

# Get the accession numbers (experiment IDs) corresponding to each of the
# samples in the feature matrix
accessions_series = store['accessions']
print(type(accessions_series))
print(accessions_series.shape)

store.close()
