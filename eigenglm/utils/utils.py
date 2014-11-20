from scipy.linalg import solve, det, inv
from numpy import prod, diag, log, einsum, diag_indices
from numpy.linalg import slogdet

def invert_low_rank(Ainv, U, C, V, diag=False):
    """
    Invert the matrix (A+UCV) where A^{-1} is known and C is lower rank than A

    Let N be rank of A and K be rank of C where K << N

    Then we can write the inverse,
    (A+UCV)^{-1} = A^{-1} - A^{-1}U (C^{-1}+VA^{-1}U)^{-1} VA^{-1}

    :param Ainv: NxN matrix A^{-1}
    :param U: NxK matrix
    :param C: KxK invertible matrix
    :param V: KxN matrix
    :return:
    """
    N,K = U.shape
    Cinv = inv(C)
    if diag:
        assert Ainv.shape == (N,)
        tmp1 = einsum('ij,j,jk->ik', V, Ainv, U)
        tmp2 = einsum('ij,j->ij', V, Ainv)
        tmp3 = solve(Cinv + tmp1, tmp2)
        # tmp4 = -U.dot(tmp3)
        tmp4 = -einsum('ij,jk->ik', U, tmp3)
        tmp4[diag_indices(N)] += 1
        return einsum('i,ij->ij', Ainv, tmp4)

    else:
        tmp = solve(Cinv + V.dot(Ainv).dot(U), V.dot(Ainv))
        return Ainv - Ainv.dot(U).dot(tmp)

def det_low_rank(Ainv, U, C, V, diag=False):
    """

    det(A+UCV) = det(C^{-1} + V A^{-1} U) det(C) det(A).

    :param Ainv: NxN
    :param U: NxK
    :param C: KxK
    :param V: KxN
    :return:
    """
    Cinv = inv(C)

    if diag:
        detA = 1.0 / prod(Ainv)
    else:
        detA = 1.0 / det(Ainv)

    return det(Cinv + V.dot(Ainv).dot(U)) * det(C) * detA

def logdet_low_rank(Ainv, U, C, V, diag=False):
    """

    logdet(A+UCV) = logdet(C^{-1} + V A^{-1} U) +  logdet(C) + logdet(A).

    :param Ainv: NxN
    :param U: NxK
    :param C: KxK
    :param V: KxN
    :return:
    """
    Cinv = inv(C)
    sC, ldC = slogdet(C)
    assert sC > 0

    if diag:
        ldA = -log(Ainv).sum()

        tmp1 = einsum('ij,j,jk->ik', V, Ainv, U)
        s1, ld1 = slogdet(Cinv + tmp1)
        assert s1 > 0

    else:
        sAinv, ldAinv = slogdet(Ainv)
        ldA = -ldAinv
        assert sAinv > 0

        s1, ld1 = slogdet(Cinv + V.dot(Ainv).dot(U))
        assert s1 > 0

    return  ld1 + ldC + ldA