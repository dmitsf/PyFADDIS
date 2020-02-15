''' FADDIS clustering implementation in Python
'''

import numpy as np
import numpy.linalg as LA

ZERO_BOUND = 10 ** (-9)
CLUSTER_CONTRIBUTION = 5 * 10 ** (-3)
EPSILON = 5 * 10 ** (-2)
# Maximum number of clusters
NUM_CLUSTERS = 15


def faddis(A):
    ''' faddis: equential extraction of fuzzy clusters, in a sequential manner

    A is NxN similatriy matrix, symmetrized
    member - NxK membership matix of clustering;
    contrib - 1xK vector of relative contributions to the data scatter;
    itensity - Kx2 matrix of cluster intensities^0.5 and intensities;
    lat - 1xK vector of eigen-values corresponding to clusters;
    cluster_got
    '''

    # minimum cluster's relative contribution to the data scatter
    conc = CLUSTER_CONTRIBUTION
    # minimum relative residual data scatter
    eps = EPSILON
    # maximum number of clusters
    numc = NUM_CLUSTERS

    is_positive = True
    matrix_dim, _ = A.shape

    sc = np.power(A, 2)
    # Total data scatter
    scatter = np.sum(sc)

    cluster_got = 0
    member = np.empty((matrix_dim, 0))
    contrib = np.array([])
    lat = np.empty((0, matrix_dim))
    intensity = np.empty((0, 2))
    cont = 1
    ep = 1

    # 'zero' and 'one' vectors for comparisons
    zerv = np.zeros((matrix_dim, 1))
    onev = np.ones((matrix_dim, 1))

    # ensure matrix is symmetrical
    At = (A + A.T) / 2
    matrix_sequence = [At]

    # Stop condition:
    # is_positive is True: eigen-value of the residual matrix is not positive;
    # OR la cluster intensity  reaches its minimum lam;
    # OR ep relative residual data scatter reaches its minimum eps;
    # OR maximum number of clusters numc is achieved
    while is_positive and cont > conc and ep > eps and cluster_got <= numc:
        # collecting a fuzzy cluster membership uf, with contrib con and intensity la,
        d, v = LA.eig(At)
        # (lt, ii) - (maximum eigen-value, corresponding position)
        da = np.diag(d)
        nonzero_cond = np.array(d > ZERO_BOUND)
        # Only positive eigenvalues
        di = np.argwhere(d > ZERO_BOUND).ravel()
        dl = di.size
        inten = np.zeros((dl, 1))
        vm = np.zeros((matrix_dim, dl))
        for k in range(dl):
            lt = da[di[k]]
            vf = v[:, di[k]]

            # Calculate normalized membership vector belonging to [0, 1] by
            # projection on the space. The normalization factor is the
            # Euclidean length of the vector
            bf = np.maximum(zerv, vf)
            uf = np.minimum(bf, onev)
            
            if LA.norm(uf) > 0:
                uf = uf / LA.norm(uf)

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

            bf1 = np.maximum(zerv, vf1)
            uf1 = np.minimum(bf1, onev)
            uf1 = np.squeeze(np.asarray(uf1))

            if LA.norm(uf1) > 0:
                uf1 = uf1 / LA.norm(uf1)
                
            vt1 = uf1.T.dot(At).dot(uf1)
            wt1 = uf1.T.dot(uf1)
            if wt1 > 0:
                la1 = vt1.item() / (wt1 **2)
            else:
                la1 = 0

            if la > la1:
                inten[k] = la
                vm[:, k] = uf.ravel()
            else:
                inten[k] = la1
                vm[:, k] = uf1.ravel()

        ite, ik = inten.max(), inten.argmax()
        if ite > ZERO_BOUND:
            lat = np.append(lat, np.matrix(da[di[ik]]), axis=0)
            intensity = np.append(intensity, np.matrix([np.sqrt(ite), ite]), axis=0)
            # square root and value of lambda intensity of cluster_got
            # square root shows the value of fuzzyness
            uf = vm[:,ik]
            vt = uf.T.dot(At).dot(uf)
            wt = uf.T.dot(uf)
            #member[:,tt]=uf
            member = np.append(member, np.matrix(uf).T, axis=1)
            # calculate residual similarity matrix:
            # remove the present cluster (i.e. itensity* membership) from
            # similarity matrix
            Att = At - ite * np.matrix(uf).T*np.matrix(uf)

            At = (Att+Att.T) / 2
            matrix_sequence.append(At)
            cont = (vt / wt) ** 2
            # Calculate the relative contribution of cluster_got
            cont /= scatter
            contrib = np.append(contrib, cont)
            # Calculate the residual contribution
            ep -= cont
            cluster_got += 1
        else:
            is_positive = False

    if not is_positive:
        print('No positive weights at spectral clusters')
    elif cont < conc:
        print('Cluster contribution is too small')
    elif ep < eps:
        print('Residual is too small')
    elif cluster_got > numc:
        print('Maximum number of clusters reached')

    return matrix_sequence, member, contrib, intensity, lat, cluster_got


if __name__ == '__main__':

    M = np.matrix([[1, .5, .3,  .1],
                   [.5, 1, .98, .4],
                   [.3, .98, 1, .6],
                   [.1, .4, .6, 1 ]])
    #M = np.matrix([[1, 0, 1], [0, 3, 0], [1, 0, 9]])
    #M = np.matrix(np.random.rand(500,500))
    B, member, contrib, intensity, lat, tt = faddis(M)
    #exit()
    print("B")
    print(B)
    print("member")
    print(member)
    print("contrib")
    print(contrib)
    print("intensity")
    print(intensity)
    print("lat")
    print(lat)
    print("tt")
    print(tt)
