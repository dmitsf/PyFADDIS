import numpy as np
import numpy.linalg as LA

ZERO_BOUND = 10 ** (-8)
ENTITY_BOUND = 10 ** (-4)


def lapin(A):
    A = (A + A.T) / 2
    a_sums = np.ravel(abs(sum(A)))
    t = np.array(a_sums > ENTITY_BOUND)
    correct = t.all()

    if not correct:
        print('These entities are no good - remove them first!!!')
        print([i for i, j in enumerate(t, 1) if not j])

    nn, nn = A.shape
    if correct:
        AR = A
        ar = a_sums
    else:
        AR = A[:, t][t, :]
        ar = a_sums[t]

    n, n = AR.shape
    D = np.eye(n)
    C = np.empty([n, n])

    ar = np.ravel(ar)
    for ii in range(n):
        for jj in range(n):
            C[ii, jj]=AR[ii, jj] / np.sqrt(ar[ii] * ar[jj])

    Bt = D - C
    Ln = Bt
    ll, vv = LA.eig(Bt)
    ld = np.diag(ll)
    cond = np.array(ll > ZERO_BOUND)

    lcd = ld[cond, :][:, cond]
    vd = vv[:, cond]
    B = vd.dot(LA.inv(lcd)).dot(vd.T)

    return B


if __name__ == '__main__':

    M = np.matrix([[1, 2, 1], [2, 4, 1], [1, 1, 9]])
    M = np.matrix([[1, 0, 1], [0, 0, 0], [1, 0, 9]])
    M_transformed = lapin(M)
    print(M_transformed)
