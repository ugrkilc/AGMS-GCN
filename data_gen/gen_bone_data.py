import os
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from tqdm import tqdm

ntu_skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),(25, 12)
)

n_ucla_skeleton_bone_pairs = [(1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
                    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (3, 3), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19)] 




bone_pairs = {
    'Ntu60/xview': ntu_skeleton_bone_pairs,
    'Ntu60/xsub': ntu_skeleton_bone_pairs,

    # NTU 120 uses the same skeleton structure as NTU 60
    'Ntu120/xsub': ntu_skeleton_bone_pairs,
    'Ntu120/xset': ntu_skeleton_bone_pairs,

    'n_ucla/': n_ucla_skeleton_bone_pairs,

    'kinetics': (
        (0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
        (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15)
    )


}

benchmarks = {
    'Ntu60': ('Ntu60/xview', 'Ntu60/xsub'),
    'Ntu120': ('Ntu120/xset', 'Ntu120/xsub'),
    'N-Ucla': ('n_ucla'),
    'kinetics': ('kinetics')
}

parts = { 'train', 'val' }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bone data generation for NTU/N-Ucla/Kinetics')
    parser.add_argument('--dataset', choices=['Ntu60', 'Ntu120', 'N-Ucla', 'kinetics'], required=True)
    args = parser.parse_args()

    for benchmark in benchmarks[args.dataset]:
        for part in parts:
            print(benchmark, part)
            data = np.load('{}/{}_data_joint.npy'.format(benchmark, part))
            N, C, T, V, M = data.shape
            fp_sp = open_memmap(
                '{}/{}_data_bone.npy'.format(benchmark, part),
                dtype='float32',
                mode='w+',
                shape=(N, 3, T, V, M))
            # Copy the joints data to bone placeholder tensor
            fp_sp[:, :C, :, :, :] = data
            for v1, v2 in tqdm(bone_pairs[benchmark]):
                # Reduce class index for NTU datasets
                if benchmark != 'kinetics':
                    v1 -= 1
                    v2 -= 1
                # Assign bones to be joint1 - joint2, the pairs are pre-determined and hardcoded
                # There also happens to be 25 bones
                fp_sp[:, :, :, v1, :] = data[:, :, :, v1, :] - data[:, :, :, v2, :]

