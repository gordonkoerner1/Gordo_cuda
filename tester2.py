import numpy as np
import scipy.sparse as sp
import cupy as cp
from Gordon_library.cusparse_module import solve_geam

def sort_csr(matrix):
    row_start = 0
    for i in range(len(matrix.indptr) - 1):
        row_end = matrix.indptr[i + 1]
        indices = matrix.indices[row_start:row_end]
        data = matrix.data[row_start:row_end]
        sorted_indices = np.argsort(indices)
        matrix.indices[row_start:row_end] = indices[sorted_indices]
        matrix.data[row_start:row_end] = data[sorted_indices]
        row_start = row_end


# Parameters
np.random.seed(69) #some random seeds produce a row singularity and break stuff
n = 1000  # Size of the matrix (adjust as needed)
density = .01 # Density of non-zero elements (adjust as needed)

# Generate a sparse symmetric positive definite matrix A
A = sp.random(n, n, density=density, format='csr') + sp.eye(n,n)
A = A.T.dot(A)  # Ensure A is symmetric and positive definite

# Convert A to cupyx if needed (for CUDA-based operations)
sort_csr(A)
A_data = cp.asarray(A.data)
A_indices = cp.asarray(A.indices)
A_indptr = cp.asarray(A.indptr)

# Generate a sparse symmetric positive definite matrix B
B = sp.random(n, n, density=density, format='csr') + sp.eye(n,n)
B = B.T.dot(B)  # Ensure B is symmetric and positive definite

# Convert A and B to cupy arrays if needed
sort_csr(B)
B_data = cp.asarray(B.data)
B_indices = cp.asarray(B.indices)
B_indptr = cp.asarray(B.indptr)

# force the cupy arrays to store data contiguously
A_data = cp.ascontiguousarray(A_data)
A_indices = cp.ascontiguousarray(A_indices)
A_indptr = cp.ascontiguousarray(A_indptr)

# force the cupy arrays to store data contiguously
B_data = cp.ascontiguousarray(B_data)
B_indices = cp.ascontiguousarray(B_indices)
B_indptr = cp.ascontiguousarray(B_indptr)

# Now you can test your solve_cholesky function with A_cupy and b_cupy
# Assuming K_active is your cupyx CSR matrix
csrRowPtrA_ptr = A_indptr.__cuda_array_interface__['data'][0]   # csrRowPtr
csrColIndA_ptr = A_indices.__cuda_array_interface__['data'][0]  # csrColInd
csrValA_ptr    = A_data.__cuda_array_interface__['data'][0]     # csrValA
nnz_A          = A_data.shape[0]
csrRowPtrB_ptr = B_indptr.__cuda_array_interface__['data'][0]   # csrRowPtr
csrColIndB_ptr = B_indices.__cuda_array_interface__['data'][0]  # csrColInd
csrValB_ptr    = B_data.__cuda_array_interface__['data'][0]     # csrValA
nnz_B          = B_data.shape[0]
rows           = A_indptr.shape[0] - 1

test=solve_geam(csrRowPtrA_ptr, csrColIndA_ptr, csrValA_ptr, nnz_A, csrRowPtrB_ptr, csrColIndB_ptr, csrValB_ptr, nnz_B, rows)

# Wrap the results of solve_geam into cupy.ndarray
nnzC = test[3]
# Example values
# Calculate the total size in bytes for the memory allocation

# Create a cupy.ndarray from the raw device pointer
C_indptr = cp.ndarray((n + 1,), dtype=cp.int32, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(test[0], cp.dtype(cp.int32).itemsize * nnzC, None), 0))

C_indices = cp.ndarray((nnzC,), dtype=cp.int32, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(test[1], cp.dtype(cp.int32).itemsize * nnzC, None), 0))

C_data = cp.ndarray((nnzC,), dtype=cp.float64, memptr=cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(test[2], cp.dtype(cp.float64).itemsize * nnzC, None), 0))


