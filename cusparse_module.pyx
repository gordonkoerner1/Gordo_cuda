# cython: language_level=3
# Include helper_cuda.h
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t
from libc.stddef cimport size_t
from libc.stdio cimport printf

# External declarations of cudart
cdef extern from "cuda_runtime_api.h":
    ctypedef int cudaError_t
    int cudaDeviceSynchronize()

    ctypedef enum cudaMemcpyKind_t:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
        cudaMemcpyDefault

    cudaError_t cudaMalloc(void** devPtr, size_t size)
    cudaError_t cudaFree(void* devPtr)

    cudaError_t cudaMemcpy(void *dst,
                           const void *src,
                           size_t count,
                           cudaMemcpyKind_t kind);

# External declarations of cuSPARSE functions and types
cdef extern from "cusparse.h":
    # Define enumerated types as in CUDA cuSPARSE
    ctypedef enum cusparseMatrixType_t:
        CUSPARSE_MATRIX_TYPE_GENERAL
        CUSPARSE_MATRIX_TYPE_SYMMETRIC
        CUSPARSE_MATRIX_TYPE_HERMITIAN
        CUSPARSE_MATRIX_TYPE_TRIANGULAR

    ctypedef enum cusparseFillMode_t:
        CUSPARSE_FILL_MODE_LOWER
        CUSPARSE_FILL_MODE_UPPER

    ctypedef enum cusparseDiagType_t:
        CUSPARSE_DIAG_TYPE_NON_UNIT
        CUSPARSE_DIAG_TYPE_UNIT

    ctypedef enum cusparseIndexBase_t:
        CUSPARSE_INDEX_BASE_ZERO
        CUSPARSE_INDEX_BASE_ONE

    ctypedef enum cusparsePointerMode_t:
        CUSPARSE_POINTER_MODE_HOST
        CUSPARSE_POINTER_MODE_DEVICE

    ctypedef void* cusparseHandle_t
    ctypedef void* cusparseMatDescr_t
    ctypedef int cusparseStatus_t

    # Function declarations // WARNING OPAQUE DATA STRUCTURES
    cusparseStatus_t cusparseCreate(cusparseHandle_t*)
    cusparseStatus_t cusparseDestroy(cusparseHandle_t)
    cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t*)
    cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t)
    cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t, cusparseMatrixType_t)
    cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t, cusparseIndexBase_t)
    cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t, cusparsePointerMode_t)
    cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t, cusparseDiagType_t)   
    cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t, cusparseFillMode_t)

    cusparseStatus_t cusparseDcsrgeam2_bufferSizeExt(
                        cusparseHandle_t,
                        int,
                        int,
                        double*,
                        cusparseMatDescr_t,
                        int,
                        double*,
                        int*,
                        int*,
                        double*,
                        cusparseMatDescr_t,
                        int,
                        double*,
                        int*,
                        int*,
                        cusparseMatDescr_t,
                        double*,
                        int*,
                        int*,
                        size_t*)

    cusparseStatus_t cusparseXcsrgeam2Nnz(
                        cusparseHandle_t,
                        int,
                        int,
                        cusparseMatDescr_t,
                        int,
                        int*,
                        int*,
                        cusparseMatDescr_t,
                        int,
                        int*,
                        int*,
                        cusparseMatDescr_t,
                        int*,
                        int*,
                        void*)
    
    cusparseStatus_t cusparseDcsrgeam2(
                        cusparseHandle_t,
                        int,
                        int,
                        double*,
                        cusparseMatDescr_t,
                        int,
                        double*,
                        int*,
                        int*,
                        double*,
                        cusparseMatDescr_t,
                        int,
                        double*,
                        int*,
                        int*,
                        cusparseMatDescr_t,
                        double*,
                        int*,
                        int*,
                        void*)

# Function to solve using cuSOLVER and cuSPARSE
@cython.boundscheck(False)  # Disable bounds-checking for faster access
@cython.wraparound(False)   # Disable negative indexing
def solve_geam(uintptr_t csrRowPtrA_ptr,
               uintptr_t csrColIndA_ptr,
               uintptr_t csrValA_ptr,
               int nnz_A,
               uintptr_t csrRowPtrB_ptr,
               uintptr_t csrColIndB_ptr,
               uintptr_t csrValB_ptr,
               int nnz_B,
               int rows):

    cdef cusparseHandle_t cusparseH = NULL
    # Allocate descriptors
    cdef cusparseMatDescr_t descrA = NULL
    cdef cusparseMatDescr_t descrB = NULL
    cdef cusparseMatDescr_t descrC = NULL
    cdef size_t bufferSize
    cdef void* pBuffer = NULL
    cdef double alpha = 1.0
    cdef double beta = 1.0
    cdef int status, nnzC, testout1, testout2, testout3
    cdef int baseC = 0 
    cdef int* nnzC_Ptr = &nnzC
    
    #INITIALIZE DUMMY <uintptr_t> inputs, they're not passed to solve_geam
    cdef int* d_csrRowPtrC = NULL
    cdef int* d_csrColIndC = NULL
    cdef double* d_csrValC = NULL
    cdef int* temp_d_csrColIndC = NULL #used before full d_csrColIndC mem allo
    cdef double* temp_d_csrValC = NULL #used before full d_csrValC mem allo
    
    #allocate for the dummies
    # Allocating memory for d_csrRowPtrC, we know "rows" so can do full allo
    cdef size_t size_allocated = sizeof(int) * (rows + 1)
    # Allocate memory for d_csrRowPtrC entirely
    status = cudaMalloc(<void**>&d_csrRowPtrC, size_allocated)
    if status != 0:
        raise MemoryError("Failed to allocate device memory for d_csrRowPtrC")
    
    # Allocate temporary memory for d_csrColIndC
    status = cudaMalloc(<void**>&d_csrColIndC, sizeof(int))
    if status != 0:
        raise RuntimeError("d_csrColIndC not initialized")
    
    # Allocate temporary memory for d_csrValC
    status = cudaMalloc(<void**>&d_csrValC, sizeof(double))
    if status != 0:
        raise RuntimeError("d_csrValC not initialized")

    # Cast pointers to the correct types
    cdef double* alpha_ptr = &alpha
    cdef double* beta_ptr = &beta

    # Cast CSR format pointers to the correct types
    cdef int* d_csrRowPtrA = <int*>csrRowPtrA_ptr
    cdef int* d_csrColIndA = <int*>csrColIndA_ptr
    cdef double* d_csrValA = <double*>csrValA_ptr
    
    cdef int* d_csrRowPtrB = <int*>csrRowPtrB_ptr
    cdef int* d_csrColIndB = <int*>csrColIndB_ptr
    cdef double* d_csrValB = <double*>csrValB_ptr
    
    status = cudaDeviceSynchronize()
    if status != 0:
        raise RuntimeError("cuda device could not synchronize")

    # Create cusparse handle
    # Initialize cuSPARSE
    status = cusparseCreate(&cusparseH)
    if status != 0:
        raise RuntimeError("cusparseCreate failed")
        
    status = cusparseSetPointerMode(cusparseH, CUSPARSE_POINTER_MODE_HOST)
    if status != 0:
        raise RuntimeError("cusparse_pointer_mode_host not set")

    # Create descriptors
    status = cusparseCreateMatDescr(&descrA)
    if status != 0:
        raise RuntimeError("cusparseCreateMatDescr descrA failed")

    status = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL)
    if status != 0:
        raise RuntimeError("cusparseSetMatType descrA failed")
    # status = cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_UPPER)
    # if status != 0:
    #     raise RuntimeError('cusparseSetMatFillMode descrA failed')
    status = cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT)
    if status != 0:
        raise RuntimeError('cusparseSetMatDiagType descrA failed')
    status = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO)
    
    # Create descriptors
    status = cusparseCreateMatDescr(&descrB)
    if status != 0:
        raise RuntimeError("cusparseCreateMatDescr descrB failed")

    status = cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL)
    if status != 0:
        raise RuntimeError("cusparseSetMatType descrB failed")
    # status = cusparseSetMatFillMode(descrB, CUSPARSE_FILL_MODE_UPPER)
    # if status != 0:
    #     raise RuntimeError('cusparseSetMatFillMode descrB failed')
    status = cusparseSetMatDiagType(descrB, CUSPARSE_DIAG_TYPE_NON_UNIT)
    if status != 0:
        raise RuntimeError('cusparseSetMatDiagType descrB failed')
    status = cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ZERO)
    if status != 0:
        raise RuntimeError("cusparseSetMatIndexBase descrB failed")
    
    # Create descriptors
    status = cusparseCreateMatDescr(&descrC)
    if status != 0:
        raise RuntimeError("cusparseCreateMatDescr descrC failed")

    status = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL)
    if status != 0:
        raise RuntimeError("cusparseSetMatType descrC failed")
    # status = cusparseSetMatFillMode(descrC, CUSPARSE_FILL_MODE_UPPER)
    # if status != 0:
    #     raise RuntimeError('cusparseSetMatFillMode descrC failed')
    status = cusparseSetMatDiagType(descrC, CUSPARSE_DIAG_TYPE_NON_UNIT)
    if status != 0:
        raise RuntimeError('cusparseSetMatDiagType descrC failed')
    status = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO)
    if status != 0:
        raise RuntimeError("cusparseSetMatIndexBase descrC failed")

    status = cudaDeviceSynchronize()
    if status != 0:
        raise RuntimeError("cuda device could not synchronize")

    # Determine buffer size
    status = cusparseDcsrgeam2_bufferSizeExt(
                    cusparseH,
                    rows, rows,
                    alpha_ptr,
                    descrA, nnz_A,
                    d_csrValA, d_csrRowPtrA, d_csrColIndA,
                    beta_ptr,
                    descrB, nnz_B,
                    d_csrValB, d_csrRowPtrB, d_csrColIndB,
                    descrC,
                    d_csrValC, d_csrRowPtrC, d_csrColIndC,
                    &bufferSize)
    if status != 0:
        raise RuntimeError("cusparseDcsrgeam2_bufferSizeExt failed with status: %d" % status)

    
    # Allocate buffer
    status = cudaMalloc(<void**>&pBuffer, sizeof(char) * bufferSize)
    if status != 0:
        raise RuntimeError("cudaMalloc failed with status: %d" % status)
        
    # # Determine nnzC (number of non-zero elements in matrix C)
    # print("Inputs to cusparseXcsrgeam2Nnz:")
    # print("Parameter                   | Value")
    # print("--------------------------------------")
    # print("cusparseH                   |", <uintptr_t>cusparseH)
    # print("rows                        |", rows)
    # print("descrA                      |", <uintptr_t>descrA)
    # print("nnz_A                       |", nnz_A)
    # print("d_csrRowPtrA                |", <uintptr_t>d_csrRowPtrA)
    # print("d_csrColIndA                |", <uintptr_t>d_csrColIndA)
    # print("descrB                      |", <uintptr_t>descrB)
    # print("nnz_B                       |", nnz_B)
    # print("d_csrRowPtrB                |", <uintptr_t>d_csrRowPtrB)
    # print("d_csrColIndB                |", <uintptr_t>d_csrColIndB)
    # print("descrC                      |", <uintptr_t>descrC)
    # print("d_csrRowPtrC                |", <uintptr_t>d_csrRowPtrC)
    # print("nnzC (pointer)              |", <uintptr_t>nnzC_Ptr)
    # print("nnzC (value)                |", nnzC)
    # print("pBuffer (pointer)           |", <uintptr_t>pBuffer)
    # print("pBuffer (value)             |", bufferSize)
    
    # Determine nnzC
    status = cusparseXcsrgeam2Nnz(cusparseH,
                                  rows, rows,
                                  descrA, nnz_A,
                                  d_csrRowPtrA, d_csrColIndA,
                                  descrB, nnz_B,
                                  d_csrRowPtrB, d_csrColIndB,
                                  descrC,
                                  d_csrRowPtrC, nnzC_Ptr,
                                  pBuffer)
    if status != 0:
        raise RuntimeError("cusparseXcsrgeam2Nnz failed with status: %d" % status)

    #assign value to nncZ
    if (NULL != nnzC_Ptr):
        nnzC = nnzC_Ptr[0]
    else:
        print('NULL != nnzC_Ptr: false') #toubleshooting
        cudaMemcpy(&nnzC, d_csrRowPtrC + rows, sizeof(int), cudaMemcpyDeviceToHost)
        cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost)
        nnzC -= baseC
        
    # print("nnzC (pointer)              |", <uintptr_t>nnzC_Ptr)
    # print("nnzC (value)                |", nnzC)
    
    # status = cudaMemcpy(&testout1, d_csrRowPtrC + rows, sizeof(int), cudaMemcpyDeviceToHost)
    # if status != 0:
    #     raise RuntimeError("cudaMemcpy failed testout1 with status: %d" % status)
        
    # status = cudaMemcpy(&testout2, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost)
    # if status != 0:
    #     raise RuntimeError("cudaMemcpy failed testout2 with status: %d" % status)

    # Allocate memory for d_csrColIndC and d_csrValC
    status = cudaMalloc(<void**>&temp_d_csrColIndC, sizeof(int) * nnzC)
    if status != 0:
        raise MemoryError("Failed to allocate device memory for temp_d_csrColIndC")
    status = cudaMalloc(<void**>&temp_d_csrValC, sizeof(double) * nnzC)
    if status != 0:
        raise MemoryError("Failed to allocate device memory for temp_d_csrValC")
    
    # Free the temporary memory blocks for these line vectors
    if d_csrColIndC is not NULL:
        cudaFree(d_csrColIndC)
    if d_csrValC is not NULL:
        cudaFree(d_csrValC)
    
    # Update pointers to new memory block
    d_csrColIndC = temp_d_csrColIndC
    d_csrValC    = temp_d_csrValC    
    
    status = cudaDeviceSynchronize()
    if status != 0:
        raise RuntimeError("cuda device could not synchronize")
    
    # Perform sparse matrix addition
    status = cusparseDcsrgeam2(cusparseH,
                               rows, rows,
                               alpha_ptr,
                               descrA, nnz_A,
                               d_csrValA, d_csrRowPtrA, d_csrColIndA,
                               beta_ptr,
                               descrB, nnz_B,
                               d_csrValB, d_csrRowPtrB, d_csrColIndB,
                               descrC,
                               d_csrValC, d_csrRowPtrC, d_csrColIndC,
                               pBuffer)
    if status != 0:
        raise RuntimeError("cusparseDcsrgeam2 failed")

      # Clean up
    if pBuffer:
        cudaFree(pBuffer)
    # When cleaning up:
    cusparseDestroyMatDescr(descrA)
    cusparseDestroyMatDescr(descrB)
    cusparseDestroyMatDescr(descrC)
    if cusparseH:
        cusparseDestroy(cusparseH)
    
    # Retrieve the integer values of the pointers
    cdef uintptr_t row_ptr_address = <uintptr_t>d_csrRowPtrC
    cdef uintptr_t col_ind_address = <uintptr_t>d_csrColIndC
    cdef uintptr_t values_address = <uintptr_t>d_csrValC
    
    # Return the addresses as their types
    return row_ptr_address, col_ind_address, values_address, nnzC
