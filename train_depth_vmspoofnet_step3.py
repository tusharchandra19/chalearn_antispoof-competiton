# -*- coding:utf-8 -*-

import sys
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, data, fit
import mxnet as mx

def get_nir_flow_data():
    data_dir="./data/"
    # fnames = (os.path.join(data_dir, "train_depth_noenmfake_112_15460.rec"),
    #           os.path.join(data_dir, "val_depth_all_112_9608.rec"))
    # fnames = (os.path.join(data_dir, "train_depth_all_112_29266.rec"),
    #           os.path.join(data_dir, "val_depth_all_112_9608.rec"))
    fnames = (os.path.join(data_dir, "train_depth_aug_112_38208.rec"),
              os.path.join(data_dir, "val_depth_all_112_9608.rec"))
    return fnames


if __name__ == '__main__':
    # download data
    (train_fname, val_fname) = get_nir_flow_data()

    # parse args
    parser = argparse.ArgumentParser(description="train-casia-surf-cefa-depth",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_fit_args(parser)
    data.add_data_args(parser)
    data.add_data_aug_args(parser)
    data.set_data_aug_level(parser, 2)
    parser.set_defaults(
        # network
        network        = 'vmspoofnet',
        #num_layers     = 110,
        # data
        data_train     = train_fname,
        data_val       = val_fname,
        num_classes    = 2,
        num_examples   = 38208,  # 训练样本数
        image_shape    = '3,224,224', # channel,height,width
        pad_size       = 0,

        # data aug
        max_random_rotate_angle = 45,
        max_random_aspect_ratio = 0.5,
        max_random_shear_ratio = 0.5,
        max_random_h = 15,
        max_random_s = 15,
        max_random_l = 15,
        # max_random_scale = 0,
        # min_random_scale = 0,
        # random_crop = 0,
        # train
        batch_size     = 512,
        num_epochs     = 1500,
        #wd             = 0.000001,
        lr             = 1e-6,
        #lr_factor      = 0.5,
        lr_step_epochs = '900',
        model_prefix   = 'checkpoint_depth_112_29266_38208_vmspoofnet_2m',
        checkpoint_period = 1, # How many epochs to wait before checkpointing. Defaults to 1.
        # checkpoint_period = 1,
	    load_epoch     = 68,
	    gpus           = '1,2,3'
    )
    args = parser.parse_args()

    # load network
    from importlib import import_module
    net = import_module('symbols.'+args.network)
    sym = net.get_symbol(**vars(args))

    # train
    fit.fit(args, sym, data.get_rec_iter)

