from __future__ import division
import numpy as np
import numpy as np

def permutation_test_tri(triu, n_1, n_2, n_permutations):
    """
    Given the upper triangular part (excluding the diagonal) part of a
    matrix, perform the permutation test over:

        \sum_{i<j} [\pi(i) == \pi(j)] a_{i,j}.

    As we reject for high p, we count how many values under the permutation
    null are larger than or equal to the computed statistic.

    Arguments:
    ----------
    triu : array of size n * (n - 1) / 2
        The upper triangular part of the matrix, row-order.
    n_1 : int
        Number of elements in the first group.
    n_2 : int
        Number of elements in the second group.
    n_permutations : int
        How many random permutations to use to estimate the p-value.
    """
    n = n_1 + n_2
    pi = np.zeros(n, dtype=np.int8)
    pi[n_1:] = 1

    statistic = 0
    larger = 0  # Number of permutations with a larger score.
    
    k = 0  # Indexing the vector `triu`.
    
    # The statistic of the sample corresponds to the permutation
    # [1] * n_1 + [0] * n_0, which we compute in the first iteration.
    for sample_n in range(n_permutations + 1):
        count = 0
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pi[i] == pi[j]:
                    count += triu[k]
                k += 1
        if sample_n == 0:  # First permutation, just save the statistic.
            statistic = count
        elif statistic <= count:
            larger += 1

        np.random.shuffle(pi)

    return larger / n_permutations

def permutation_test_mat(matrix, n_1, n_2, n_permutations, a00=1, a11=1, a01=0):
    """
    Compute the p-value of the following statistic (rejects when high):

        \sum_{i,j} a_{\pi(i), \pi(j)} matrix[i, j].
    """
    n = n_1 + n_2
    pi = np.zeros(n, dtype=np.int8)
    pi[n_1:] = 1

    statistic = 0
    larger = 0

    for sample_n in range(n_permutations + 1):
        count = 0
        for i in range(n):
            for j in range(i, n):
                mij = matrix[i, j] + matrix[j, i]
                if pi[i] == pi[j] == 0:
                    count += a00 * mij
                elif pi[i] == pi[j] == 1:
                    count += a11 * mij
                else:
                    count += a01 * mij
        if sample_n == 0:
            statistic = count
        elif statistic <= count:
            larger += 1

        np.random.shuffle(pi)

    return larger / n_permutations
