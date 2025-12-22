import numpy as np
import time
# =============================================================================
# Operator utilities
# =============================================================================

def anni (dim:int) -> np.ndarray :
    '''
    function that creates an arbitrary dimension annihilation operator.

    Args:
    dim (:int): the dimension of the hilberst space on which anni acts.

    returns:
    a (:np.array(np.complex128)): A dim*dim matrix representation of a
    dimension dim Hilbert space annihilation operator.
    
    '''

    # define a zero matrix with the dimensions dimxdim
    # complex128 gives us 16 decimal places so that 
    # rounding errors don't stack


    a = np.zeros((dim,dim), dtype=np.complex128)
    i = np.arange(dim - 1) 
            # np.arrange(), and np.sqrt() are much faster
            # since they avoid python loop overhead and run in C
    a[i, i + 1] = np.sqrt(i + 1)
        
    return a

# =============================================================================


