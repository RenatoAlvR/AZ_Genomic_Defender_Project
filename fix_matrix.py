import scipy.io
import scipy.sparse
import gzip
import os

# Paths
input_mtx = "/home/lpr23/renato/GenomeDefender/data/GSE161529/possible_cancer/mix_neoplastic1/matrix.mtx.gz"
output_mtx = "/home/lpr23/renato/GenomeDefender/data/GSE161529/possible_cancer/mix_neoplastic1/matrix_corrected.mtx"

# Read the original matrix (as compressed MTX)
with gzip.open(input_mtx, "rt") as f:
    matrix = scipy.io.mmread(f)
print(f"Original matrix shape: {matrix.shape}")  # Should print (69032, 33538)

# Transpose to (genes, cells) = (33538, 69032)
matrix_transposed = matrix.T
print(f"Transposed matrix shape: {matrix_transposed.shape}")

# Convert to COO for easy index access if not already
if not scipy.sparse.isspmatrix_coo(matrix_transposed):
    matrix_transposed = matrix_transposed.tocoo()

# Write the transposed matrix in MTX format
with open(output_mtx, "w") as f:
    f.write("%%MatrixMarket matrix coordinate real general\n%\n")
    f.write(f"{matrix_transposed.shape[0]} {matrix_transposed.shape[1]} {matrix_transposed.nnz}\n")
    for i, j, v in zip(matrix_transposed.row, matrix_transposed.col, matrix_transposed.data):
        f.write(f"{i+1} {j+1} {int(v) if v.is_integer() else v}\n")  # Handle integer/float values

# Compress the new file
os.system(f"gzip {output_mtx}")
os.rename(f"{output_mtx}.gz", input_mtx)  # Overwrite the original matrix.mtx.gz

print("Matrix transposed and saved successfully.")
