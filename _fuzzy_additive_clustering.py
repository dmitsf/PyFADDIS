"""Fuzzy additive clustering (FADDIS)."""

import numpy as np

import scipy.linalg import eig, norm

ZERO_BOUND = 10 ** (-9)
MIN_CLUSTER_CONTRIBUTION = 5 * 10 ** (-3)
EPSILON = 5 * 10 ** (-2)
# Maximum number of clusters
MAX_NUM_CLUSTERS = 15


ZERO_BOUND = 10 ** (-8)
ENTITY_BOUND = 10 ** (-4)


def lapin(A):
    A = (A + A.T) / 2
    a_sums = np.ravel(abs(sum(A)))
    checked = np.array(a_sums > ENTITY_BOUND)
    is_correct = checked.all()

    if not is_correct:
        print('These entities are no good - remove them first!!!')
        print([i for i, j in enumerate(checked, 1) if not j])
        A = A[:, checked][checked, :]
        a_sums = a_sums[checked]

    matrix_dim, _ = A.shape
    C = np.empty((matrix_dim, matrix_dim))
    for i in range(matrix_dim):
        for j in range(matrix_dim):
            C[i, j] = A[i, j] / np.sqrt(a_sums[i] * a_sums[j])

    eig_vals, eig_vecs = LA.eig(np.eye(matrix_dim) - C)
    eig_vals_diag = np.diag(eig_vals)
    nonzero_cond = np.array(eig_vals > ZERO_BOUND)
    nonzero_eig_vals_diag = eig_vals_diag[nonzero_cond, :][:, nonzero_cond]
    nonzero_eig_vecs = eig_vecs[:, nonzero_cond]
    B = nonzero_eig_vecs.dot(LA.inv(nonzero_eig_vals_diag)).dot(nonzero_eig_vecs.T)

    return B


def faddis(A):
    ''' faddis: equential extraction of fuzzy clusters, in a sequential manner

    A is NxN similatriy matrix, symmetrized
    membership_matrix - NxK membership matix of clustering;
    contrib - 1xK vector of relative contributions to the data scatter;
    intensity - Kx2 matrix of cluster intensities^0.5 and intensities;
    lat - 1xK vector of eigen-values corresponding to clusters;
    cluster_got
    '''

    # minimum cluster's relative contribution to the data scatter
    min_cont = MIN_CLUSTER_CONTRIBUTION
    # minimum relative residual data scatter
    eps = EPSILON
    # maximum number of clusters
    max_clust_num = MAX_NUM_CLUSTERS

    is_positive = True
    matrix_dim, _ = A.shape

    sc = np.power(A, 2)
    # Total data scatter
    scatter = np.sum(sc)

    cluster_got = 0
    membership_matrix = np.empty((matrix_dim, 0))
    contrib = np.array([])
    lat = np.array([])
    intensities = np.empty((0, 2))
    curr_cont = 1
    res_cont = 1

    # 'zero' and 'one' vectors for comparisons
    zeros_vect = np.zeros((matrix_dim, 1))
    ones_vect = np.ones((matrix_dim, 1))

    # ensure matrix is symmetrical
    At = (A + A.T) / 2
    matrix_sequence = [At]

    # Stop condition:
    # is_positive is True: eigen-value of the residual matrix is not positive;
    # OR la cluster intensity  reaches its minimum lam;
    # OR ep relative residual data scatter reaches its minimum eps;
    # OR maximum number of clusters max_clust_num is achieved
    while is_positive and curr_cont > min_cont and res_cont > eps and cluster_got <= max_clust_num:
        # collecting a fuzzy cluster membership uf, with contrib con and intensity la,
        eig_vals, eig_vecs = eig(At)
        # (lt, ii) - (maximum eigen-value, corresponding position)
        eig_vals_diag = np.diag(eig_vals)
        # Only positive eigenvalues
        eig_vals_pos = np.argwhere(eig_vals > ZERO_BOUND).ravel()
        eig_vals_pos_len = eig_vals_pos.size
        cur_intensities = np.zeros((eig_vals_pos_len, 1))
        vm = np.zeros((matrix_dim, eig_vals_pos_len))
        for k in range(eig_vals_pos_len):
            lt = eig_vals_diag[eig_vals_pos[k]]
            vf = eig_vecs[:, eig_vals_pos[k]]

            # Calculate normalized membership vector belonging to [0, 1] by
            # projection on the space. The normalization factor is the
            # Euclidean length of the vector
            bf = np.maximum(zeros_vect, vf)
            uf = np.minimum(bf, ones_vect)
            
            if norm(uf) > 0:
                uf = uf / norm(uf)

            vt = uf.T.dot(At).dot(uf)
            uf = np.squeeze(np.asarray(uf))

            wt = uf.T.dot(uf)
            # Calculates the intensity Lambda (la) of the cluster, which is
            # defined almost as the Rayleigh quotient
            if wt > 0:
                la = vt.item() / (wt **2)
            else:
                la = 0

            # since lt*vf =(-lt)*(-vf), try symmetric version 
            # using -vf:
            vf1 = -vf

            bf1 = np.maximum(zeros_vect, vf1)
            uf1 = np.minimum(bf1, ones_vect)
            uf1 = np.squeeze(np.asarray(uf1))

            if norm(uf1) > 0:
                uf1 = uf1 / norm(uf1)

            vt1 = uf1.T.dot(At).dot(uf1)
            wt1 = uf1.T.dot(uf1)
            if wt1 > 0:
                la1 = vt1.item() / (wt1 **2)
            else:
                la1 = 0

            if la > la1:
                cur_intensities[k] = la
                vm[:, k] = uf.ravel()
            else:
                cur_intensities[k] = la1
                vm[:, k] = uf1.ravel()

        # contribs
        inten_max, inten_max_index = cur_intensities.max(), cur_intensities.argmax()
        print(inten_max_index)
        if inten_max > ZERO_BOUND:
            lat = np.append(lat, eig_vals[eig_vals_pos[inten_max_index]])
            intensities = np.append(intensities, np.matrix([np.sqrt(inten_max),
                                                            inten_max]), axis=0)
            # square root and value of lambda intensity of cluster_got
            # square root shows the value of fuzzyness
            uf = vm[:, inten_max_index]
            vt = uf.T.dot(At).dot(uf)
            wt = uf.T.dot(uf)

            membership_matrix = np.append(membership_matrix, np.matrix(uf).T, axis=1)
            # calculate residual similarity matrix:
            # remove the present cluster (i.e. itensity* membership) from
            # similarity matrix
            Att = At - inten_max * np.matrix(uf).T * np.matrix(uf)
            At = (Att + Att.T) / 2
            matrix_sequence.append(At)

            curr_cont = (vt / wt) ** 2
            # Calculate the relative contribution of cluster_got
            curr_cont /= scatter
            contrib = np.append(contrib, curr_cont)
            # Calculate the residual contribution
            res_cont -= curr_cont
            cluster_got += 1
        else:
            is_positive = False

    if not is_positive:
        print('No positive weights at spectral clusters')
    elif curr_cont < min_cont:
        print('Cluster contribution is too small')
    elif res_cont < eps:
        print('Residual is too small')
    elif cluster_got > max_clust_num:
        print('Maximum number of clusters reached')

    return matrix_sequence, membership_matrix, contrib, intensities, lat, cluster_got


if __name__ == '__main__':

    M = np.matrix([[1, .5, .3,  .1],
                   [.5, 1, .98, .4],
                   [.3, .98, 1, .6],
                   [.1, .4, .6, 1 ]])
    #M = np.matrix([[1, 0, 1], [0, 3, 0], [1, 0, 9]])
    #M = np.matrix(np.random.rand(1500,1500))
    s = []
    with open("tests.out") as f:
        for i in f:
            s.append(list(map(float, i.split())))

    M = np.matrix(s)
    #from lapin import lapin
    #M = np.matrix(lapin(M))
    
    B, member, contrib, intensity, lat, tt = faddis(M)
    print("B")
    #print(B)
    print("member")
    print(member[:10])
    print("contrib")
    print(contrib)
    print("intensity")
    print(intensity)
    print("lat")
    print(lat)
    print("tt")
    print(tt)
