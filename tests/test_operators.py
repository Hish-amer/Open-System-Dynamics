import numpy as np
import numpy.testing as npt


# =============================================================================
# Tests for operator utilities
# =============================================================================

# ========================= anni fn tests =====================================

from open_sys_dynamics.ladder_algebra.operators import anni


def test_anni_dim_1_is_zero():
    '''
    This function checks that the anni fn returns an all zero 1x1 matrix with
    anni(1).
    
    '''
    a = anni(1)
    expected = np.array([[0.0 + 0.0j]], dtype=np.complex128)
        # this is the expected behavior we know so we add it exactly and ensure 
        # the types match, since this can be complex in nature most generally.
        # although it is real, due to interactions with 1j or numerical 
        # rounding it is safest to keep them complex in quantum settings !
        # NB: complex128, stores float64 for real and float64 for 
        # imaginary part.

    npt.assert_allclose(
        a,
            # actual value the function outputs
        expected,
            # the expected value, i.e. the ideal one we want to ensure is 
            # returned.

        rtol=0,
            # relative error, so use usually with widely varying numbers or 
            # if you expect numbers to be very large etc.
            # ex. if expected = 1000, and  computed = 1000.001, then 
            # relative error is 0.001/1000 = 1e-6
            # So rtol=1e-6 would pass even though the
            # absolute difference isn’t tiny.
        atol=0
            # means “Allow an error up to this fixed amount.”
    )
        # Raises an AssertionError if two objects are not equal up to desired
        # tolerance. For exact constructs we can usually safely exact comparison 
        # so atol=0 because it’s built from exact sqrt call.

#-----------------------------------------------------------------------------#

def test_anni_dim_3_matches_known_matrix():
    '''
    This function checks if anni for dim=3 works fine.
    '''
    a = anni(3)
        
    expected = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, np.sqrt(2.0)],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    npt.assert_allclose(a, expected, rtol=0, atol=0)

#-----------------------------------------------------------------------------#

def test_anni_action_on_basis_states():
    dim = 6
    a = anni(dim)
    '''
    checking that anni is doing the lowering operation correctly, 
    so a|1> = |0>, a|2> = sqrt(2)|1>
    Check definition: a|n> = sqrt(n)|n-1> for n>=1, and a|0>=0
    '''
    e0 = np.zeros(dim, dtype=np.complex128)
    e0[0] = 1.0
        # this is the first computational basis state, so we start with an
        # all zeros vector and just replace the first entry with 1.
    
    npt.assert_allclose(
        a @ e0,
        0.0,
            # it should annihilate the |0> to all zeros vector !
            # NumPy broadcasts, meaning if prompted the way we do here, it
            # compares a scalar against a vector elementwise.
            # Asserting a vector equals 0.0 is prompted as 
            # “every element is zero.”
        rtol=0,
        atol=0
    )

    # now we repeat for the other computational statevectors, which 
    # is why e start from range 1.

    for n in range(1, dim):
        en = np.zeros(dim, dtype=np.complex128)
        en[n] = 1.0

        expected = np.zeros(dim, dtype=np.complex128)
        expected[n - 1] = np.sqrt(float(n))

        npt.assert_allclose(
            a @ en,
            expected,
            rtol=0,
            atol=0
        )

#-----------------------------------------------------------------------------#

def test_number_operator_diagonal_is_n():
    dim = 7
    a = anni(dim)
    N = a.conj().T @ a  
        # number operator is a^{dagger}*a = N
        # we want to check that our anni function generates the right N.

    expected_diag = np.arange(dim, dtype=np.float64)
        # this list goes from zero to dim-1,
        # so the number operator should have 
        # these on its diagonal. Since everything else
        # is zero it's just easier to check/test for the diagonal vals first.

    diag = np.diag(N)
        # extracts the diagonal entries of this matrix

    npt.assert_allclose(
        diag,
        expected_diag.astype(np.complex128),
        rtol=0,
        atol=1e-12,
        err_msg="Number operator diag vals are wrong (expected 0..dim-1)."
    )

    # asset real diagonal vals for the number operator.
    npt.assert_allclose(
        np.imag(diag),
        0.0,
        rtol=0,
        atol=1e-12,
        err_msg="Error: Number operator diag vals are not real (expected: real)."
    )

    # Off-diagonals should all be ~0 up to tolerances.
    off = N - np.diag(np.diag(N))
    npt.assert_allclose(
        off,
        0.0,
        rtol=0,
        atol=1e-12,
        err_msg="Number operator off-diag vals are wrong (expected all zeros)."
    )

    #-----------------------------------------------------------------------------#