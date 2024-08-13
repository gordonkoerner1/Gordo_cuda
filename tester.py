import numpy as np
import scipy.sparse as sp
from scipy.sparse import diags
import cupy as cp
import cupyx.scipy.sparse.linalg as cupy_linalg
from Gordo_cuda.cusolver_module import solve_cholesky

# Parameters
np.random.seed(69420) #some random seeds produce a row singularity and break stuff
n = 10000  # Size of the matrix (adjust as needed)
density = 0.001  # Density of non-zero elements (adjust as needed)

# Generate a sparse symmetric positive definite matrix A
A = sp.random(n, n, density=density, format='csr')
A += diags([1]*n, 0, format='csr')
A = A.T.dot(A)  # Ensure A is symmetric and positive definite

# Generate a random vector b
b = np.random.rand(n)

# Convert A to cupyx if needed (for CUDA-based operations)
A_data = cp.asarray(A.data)
A_indices = cp.asarray(A.indices)
A_indptr = cp.asarray(A.indptr)

# force the cupy arrays to store data contiguously
A_data = cp.ascontiguousarray(A_data)
A_indices = cp.ascontiguousarray(A_indices)
A_indptr = cp.ascontiguousarray(A_indptr)

# Optionally, convert b to cupy if needed
b_cupy = cp.asarray(b)
x_cupy = cp.zeros(b_cupy.shape)

b_cupy = cp.ascontiguousarray(b_cupy)
x_cupy = cp.ascontiguousarray(x_cupy)

# Now you can test your solve_cholesky function with A_cupy and b_cupy
# Assuming K_active is your cupyx CSR matrix
csrRowPtr_ptr = A_indptr.__cuda_array_interface__['data'][0]   # csrRowPtr
csrColInd_ptr = A_indices.__cuda_array_interface__['data'][0]  # csrColInd
csrVal_ptr    = A_data.__cuda_array_interface__['data'][0]     # csrValA
b_ptr         = b_cupy.__cuda_array_interface__['data'][0]
x_ptr         = x_cupy.__cuda_array_interface__['data'][0]
rows          = A_indptr.shape[0] - 1
nnz           = A_data.shape[0]

#solve it
solve_cholesky(csrRowPtr_ptr, csrColInd_ptr, csrVal_ptr, b_ptr, x_ptr, rows, nnz)

#define the sparse matrix A_cupy for testing
A_cupy = cp.sparse.csr_matrix(A)

# Define the matvec function for matrix-vector multiplication A_cupy * b_cupy
def matvec(v):
    return A_cupy.dot(v)

# Define the shape of the matrix (M, N) where A_cupy is M x N
M, N = A_cupy.shape

# Create the LinearOperator
linear_op = cupy_linalg.LinearOperator(shape=(M, N), matvec=matvec, dtype=A_cupy.dtype)

# Now you can use linear_op to perform matrix-vector multiplication
result = linear_op @ x_cupy  # Equivalent to A_cupy.dot(x_cupy)

print(A_cupy.indptr[-2:])  # Should show the start of the last row and the end of data
print(A_cupy.indices[-10:])  # Last few column indices
print(A_cupy.data[-10:])  # Last few non-zero values
