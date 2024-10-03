import os
import numpy as np

sets = {
    'train', 'val'
}

# 'Ntu60/xview', 'Ntu60/xsub' 
# 'Ntu120/xview', 'Ntu120/xsub'
# 'n_ucla'
# 'kinetics'

datasets = {
    'Ntu60/xview', 'Ntu60/xsub'
}

for dataset in datasets:
    for set in sets:
        print(dataset, set)
        data_jpt = np.load('{}/{}_data_joint.npy'.format(dataset, set))
        data_bone = np.load('{}/{}_data_bone.npy'.format(dataset, set))
        N, C, T, V, M = data_jpt.shape
        data_jpt_bone = np.concatenate((data_jpt, data_bone), axis=1)
        np.save('{}/{}_data_joint_bone.npy'.format(dataset, set), data_jpt_bone)
