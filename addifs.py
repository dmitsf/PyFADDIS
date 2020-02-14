import numpy as np
import numpy.linalg as LA

ZERO_BOUND = 10 ** (-8)
ENTITY_BOUND = 10 ** (-4)


def faddis(A):

    conc = 0.005
    eps=0.05

    numc=15     #   maximum number of clusters

    flagg=1;
    nn, _ = A.shape

    sc = np.power(A, 2);

    scatter = np.sum(sc)
    member=np.zeros((nn, 1))
    tt=1
    At=(A+A.T)/2
    B = []
    contrib = []
    B.append(np.copy(At))
    la=1
    cont=1
    ep=1
    num=0
    zerv=np.zeros((nn,1))
    onev=np.ones((nn,1))

    # Stop condition: 
    # flagg= eigen-value of the residual matrix is not positive;
    # OR la cluster intensity  reaches its minimum lam;
    # OR ep relative residual data scatter reaches its minimum eps;
    # OR maximum number of clusters numc is achieved
    while (flagg and cont > conc and ep > eps and tt <= numc):
        #collecting a fuzzy cluster membership uf, with contrib con and intensity la,
        d, v = LA.eig(At)
        print('iteration ', tt)
        # (lt, ii) - (maximum eigen-value, corresponding position)
        da = np.diag(d)
        nonzero_cond = np.array(v > ZERO_BOUND)
        di = da[nonzero_cond, :][:, nonzero_cond]
        
        dl, _ = di.shape
        inten = np.zeros((dl,1))
        vm = np.zeros((nn,dl))
        for k in range(dl):
            lt = da[di[k]]
            vf = v[:, di[k]]
            # Calculate normalized membership vector belonging to [0, 1] by
            # projection on the space. The normalization factor is the
            # Euclidean length of the vector
            bf = max (zerv, vf)
            uf = min(bf, onev)
            if LA.norm(uf) > 0:
                uf = uf / LA.norm(uf)

            vt = uf.T.dot(At).dot(uf)
            wt = uf.T.dot(uf)
            # Calculates the intensity Lambda (la) of the cluster, which is
            # defined almost as the Rayleigh quotient
            if wt > 0:
                la = vt / (wt**2)
            else:
                la = 0

            # since lt*vf =(-lt)*(-vf), try symmetric version 
            # using -vf:
            vf1 = -vf
            bf1 = max(zerv, vf1)
            uf1 = min(bf1, onev)
            if LA.norm(uf1)>0:
                uf1 = uf1 / LA.norm(uf1)

            vt1 = uf1.T.dot(At).dot(uf1)
            wt1 = uf1.T.dot(uf1)
            if wt1 > 0:
                la1 = vt1 / (wt1 **2)
            else:
                la1 = 0

            if la>la1:
                inten[k] = la
                vm[:,k] = uf
            else:
                inten[k] = la1
                vm[:,k] = uf1

        [ite,ik] = inten.max(), inten.argmax()

        if ite > ZERO_BOUND:
            lat[tt] = da[di[ik]]
            intensity(tt,:)=[sqrt(ite) ite];
            # square root and value of lambda intensity of cluster tt
            # square root shows the value of fuzzyness
            uf=vm[:,ik]
            vt=uf.T.dot(At).dot(uf)
            wt=uf.T.dot(uf)
            member[:,tt]=uf
            # calculate residual similarity matrix: remove the present cluster (i.e. itensity* membership) from
            # similarity matrix
            Att = At - ite.dot(uf).dot(uf.T)
            At=(Att+Att.T) / 2
            B.append(At)
            cont = (vt / wt) ** 2
            # Calculate the relative contribution of cluster tt
            cont /= scatter
            contrib.append(cont)
            # Calculate the residual contribution
            ep -= cont
            tt += 1
        else:
            flagg=0

    #member=member(:,tt);
    if not flagg:
        print('No positive weights at spectral clusters')
    elif(cont<conc):
        print('Cluster contribution is too small')
    elif ((ep>eps)==0):
        print('Residual is too small')
    elif (tt<=numc)==0):
        print('Maximum number of clusters reached')

    return member


if __name__ == '__main__':

    M = np.matrix([[1, 2, 1], [2, 4, 1], [1, 1, 9]])
    M = np.matrix([[1, 0, 1], [0, 3, 0], [1, 0, 9]])
    clusters = faddis(M)
    print(clusters)
