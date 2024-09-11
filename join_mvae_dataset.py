"""
Join files, cropping to correct length
- joint_pos.npy
- joint_rot.npy
(optionally)
- joint_rotmat.npy
- joint_quat.npy

Lengths: manifest.json
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='Training/vid2player3d/tennis_data', help='Path to the directory containing the npy files')
    args = parser.parse_args()


    manifest_dir = os.path.join(args.dir, 'manifest.json')
    manifest = json.loads(open(manifest_dir).read())
    seqs = manifest[0]['sequences']['fg']
    lengths = [seq['length'] for seq in seqs]

    joint_pos = np.zeros((sum(lengths), 72))
    joint_rot = np.zeros((sum(lengths), 72))
    joint_rotmat = np.zeros((sum(lengths), 24 * 3 * 3))
    joint_quat = np.zeros((sum(lengths), 24 * 4))
    print(joint_pos.shape)

    longest = max(lengths)
    longest_idx = lengths.index(longest)


    for i, seq in enumerate(seqs):
        print(seq['clip'])
        prefix = "file_res_" + seq['clip']
        joint_pos_path = os.path.join(args.dir, prefix + '_joint_pos.npy')
        joint_rot_path = os.path.join(args.dir, prefix + '_joint_rot.npy')
        joint_rotmat_path = os.path.join(args.dir, prefix + '_joint_rotmat.npy')
        joint_quat_path = os.path.join(args.dir, prefix + '_joint_quat.npy')

        joint_pos_in = np.load(joint_pos_path) # (n_frames, 72)
        joint_rot_in = np.load(joint_rot_path) # (n_frames, 72) 
        joint_rotmat_in = np.load(joint_rotmat_path)
        joint_quat_in = np.load(joint_quat_path) 


        start = sum(lengths[:i])
        end = sum(lengths[:i+1]) 

        length = min(len(joint_pos_in), lengths[i])
        
        joint_pos[start:end] = joint_pos_in[:length] 
        joint_rot[start:end] = joint_rot_in[:length]
        joint_rotmat[start:end] = joint_rotmat_in[:lengths[i]]
        joint_quat[start:end] = joint_quat_in[:lengths[i]]

        print("Done", i)

    print(joint_pos.shape)
    print(joint_rot.shape)
    print(joint_rotmat.shape)
    print(joint_quat.shape)


    np.save(os.path.join(args.dir,'joint_pos.npy'), joint_pos)
    np.save(os.path.join(args.dir,'joint_rot.npy'), joint_rot)
    np.save(os.path.join(args.dir,'joint_rotmat.npy'), joint_rotmat)
    np.save(os.path.join(args.dir,'joint_quat.npy'), joint_quat)

if __name__ == '__main__':
    main()
