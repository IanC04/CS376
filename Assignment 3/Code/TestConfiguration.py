import keypoints
import cameracali
import numpy as np
from itertools import permutations

if __name__ == "__main__":
    two_d, three_d = keypoints.getAllKeypoints()
    associated = np.row_stack((two_d, three_d)).T
    associated = associated.tolist()
    perms = permutations(associated, r=10)
    for i in perms:
        print(i)
        j = np.array(i).T
        two_d, three_d = j[:2], j[2:]
        P = cameracali.getPiMatrix(two_d, three_d)
        K, R, t = cameracali.decomposePiMatrix(P)
        if K is None or R is None or t is None:
            continue
        # TODO: Test configurations and rate based on some score
        print(P)

