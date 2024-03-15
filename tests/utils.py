from scipy.sparse import csr_array


def sparse_equal(arr_1: csr_array, arr_2: csr_array) -> bool:
    return (arr_1 != arr_2).nnz == 0
