import numpy as np
from lapin import lapin
from faddis import faddis

from operator import itemgetter

NUM_EL = 15


if __name__ == "__main__":
    relevance_matrix = np.loadtxt("relevance_matrix.txt")
    print(relevance_matrix.shape)
    tc = relevance_matrix.dot(relevance_matrix.T)
    print(tc.shape)

    tc_transformed = lapin(tc)
    B, member, contrib, intensity, lat, tt = faddis(tc_transformed)

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

    np.savetxt("clusters.dat", member)

    with open("taxonomy_leaves_fixed.txt") as fn:
        annotations = [l.strip() for l in fn]

    for cluster in member.T:
        print(list(sorted(zip(annotations, cluster.flat),
                          key=itemgetter(1), reverse=True))[:NUM_EL])
