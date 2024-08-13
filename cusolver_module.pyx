# cython: language_level=3
from libc.stdlib cimport malloc
cimport cython

# External declarations of cuSOLVER functions
cdef extern from "cusolverSp.h":
    ctypedef void* cusolverSpHandle_t
    ctypedef void* cusparseMatDescr_t
    int cusolverSpCreate(cusolverSpHandle_t*)
    int cusolverSpDestroy(cusolverSpHandle_t)
    int cusparseCreateMatDescr(cusparseMatDescr_t*)
    int cusparseDestroyMatDescr(cusparseMatDescr_t)
    int cusparseSetMatType(cusparseMatDescr_t, int)
    int cusparseSetMatIndexBase(cusparseMatDescr_t, int)
    int cusolverSpDcsrlsvchol(cusolverSpHandle_t, int, int, cusparseMatDescr_t,
                              double*, int*, int*, double*,
                              double, int, double*, int*)

# External declarations of cuSPARSE functions and types
cdef extern from "cusparse.h":
    ctypedef int cusparseStatus_t
    ctypedef int cusparseMatrixType_t
    ctypedef int cusparseIndexBase_t
    int CUSPARSE_MATRIX_TYPE_GENERAL
    int CUSPARSE_INDEX_BASE_ZERO

# Include CUDA Runtime API for device synchronization
cdef extern from "cuda_runtime_api.h":
    int cudaDeviceSynchronize()

# Function to solve using cuSOLVER and cuSPARSE
@cython.boundscheck(False)  # Disable bounds-checking for faster access
@cython.wraparound(False)   # Disable negative indexing
def solve_cholesky(size_t csrRowPtrA_ptr,
                   size_t csrColIndA_ptr,
                   size_t csrValA_ptr,
                   size_t b_ptr,
                   size_t x_ptr,
                   int rows, int nnz):
    cdef cusolverSpHandle_t cusolverH
    cdef cusparseMatDescr_t descrA
    cdef int reorder = 0
    cdef int singularity = -1  # Initialize with a negative value
    cdef double tol = 1e-14

    cusolverSpCreate(&cusolverH)
    cusparseCreateMatDescr(&descrA)
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)

    # Cast pointers to the correct types
    cdef int* d_csrRowPtrA = <int*>csrRowPtrA_ptr
    cdef int* d_csrColIndA = <int*>csrColIndA_ptr
    cdef double* d_csrValA = <double*>csrValA_ptr
    cdef double* d_b = <double*>b_ptr
    cdef double* d_x = <double*>x_ptr

    cudaDeviceSynchronize()  # Synchronize device before starting computation
    # Error checking code here

    if cusolverSpDcsrlsvchol(cusolverH,
                          rows,
                          nnz,
                          descrA,
                          d_csrValA,
                          d_csrRowPtrA,
                          d_csrColIndA,
                          d_b,
                          tol,
                          reorder,
                          d_x,
                          &singularity) !=0:
        raise ValueError("CUSOLVER failed")

    cudaDeviceSynchronize()  # Synchronize after computation to ensure all data is written back

    if singularity >= 0:
        raise ValueError(f"A is singular at row {singularity}")

    cusparseDestroyMatDescr(descrA)
    cusolverSpDestroy(cusolverH)