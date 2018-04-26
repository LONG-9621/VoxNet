#!/usr/bin/python3.5
# -*- coding: utf-8 -*-

""" Simple example for loading object binary data. """

import numpy as np
import tensorflow as tf
import os

import utils.data_helper as helper
import utils.visualization as viewer

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', './data/datasets/sydney-urban-objects-dataset',
    """directory that stores the Sydney Urban Object Dataset, short for SUOD.""")
tf.app.flags.DEFINE_integer(
    'fold', 0,
    """which fold, 0..3, for SUOD.""")
tf.app.flags.DEFINE_bool(
    'visualize', False,
    """visualize preprocess voxelization.""")
tf.app.flags.DEFINE_string(
    'npy_dir', './data/datasets/npy_generated',
    """directory to stores the SUOD preprocess results, including occupancy grid and label.""")
tf.app.flags.DEFINE_bool(
    'clear_cache', False,
    """clear previous generated preprocess results.""")
tf.app.flags.DEFINE_string(
    'type', 'training',
    """type of SUOD preprocess results, training set or testing set.""")

if __name__=='__main__':
    # delete old generated npy
    if FLAGS.clear_cache:
        if tf.gfile.Exists(FLAGS.npy_dir):
            tf.gfile.DeleteRecursively(FLAGS.npy_dir)
        tf.gfile.MakeDirs(FLAGS.npy_dir)

    dataset_file = FLAGS.dataset_dir + '/folds/fold{}.txt'.format(FLAGS.fold)

    with open(dataset_file) as f:
        save_dir = FLAGS.npy_dir + "/" + FLAGS.type
        data_dir = FLAGS.dataset_dir + "/objects/"
        file_list = f.readlines()
        for file in file_list:
            file_name = file.split('\n')[0]
            file_path = data_dir + file_name
            print('processsing {}'.format(file_name))

            cloud = helper.load_points_from_bin(file_path)

            aug_steps = 12
            cloud_list = helper.aug_data(cloud, aug_steps)

            # save pre-process pointcloud voxel
            idx = 0
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            for points in cloud_list:
                voxels, inside_points = \
                    helper.voxelize(points, voxel_size=(24,24,24), padding_size=(32,32,32), resolution=0.1)

                if FLAGS.visualize:
                    viewer.plot3DVoxel(voxels)

                # save pointcloud to *.pcd
                # if P.shape[0]>0:
                #     cloud=pcl.PointCloud(P) # P should be type: float
                #     # pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))
                #     #pcl.save(cloud,'./pcd/{}.pcd'.format(file_name.split('.bin')[0]))

                if inside_points.shape[0] > 0:
                    save_name = '{}/{}_{}.npy'.format(save_dir, file_name.split('.bin')[0], idx)
                    np.save(save_name, voxels)

                    print('saved. {}'.format(save_name))
                    idx += 1
