import numpy as np


def lapin(A):
    A = (A + A.T) / 2

    ac = abs(sum(A))
    t = ac[np.where(ac < 0.001)]

    if t:
        print('These entities are no good - remove them first!!!')
        print(t)

    (nn, nn) = A.shape

    
if __name__ == '__main__':
    print()
