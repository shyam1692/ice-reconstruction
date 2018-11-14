#!/usr/bin/env python3
"""
Created on Wed Oct 31 20:17:58 2018

@author: Shyam

Extracting features
"""

from __future__ import print_function
import os
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

#sys.path.insert(0, '../../')

from collections import OrderedDict


def parse_args(parser):
    parser.add_argument('--data_root', default='../data', type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--train_session_set', default=data_info['train_session_set'], type=list)
    parser.add_argument('--test_session_set', default=data_info['test_session_set'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    return parser.parse_args()

data_info = OrderedDict()
data_info['train_session_set'] = [
    'Data_20140325_05_001',
    #  'Data_20140325_05_002',
    #  'Data_20140325_06_001',
    'Data_20140325_07_001',
    'Data_20140325_07_002',
    'Data_20140325_07_003',
    'Data_20140325_07_004',
    #  'Data_20140325_07_005',
    'Data_20140401_03_001',
    'Data_20140401_03_002',
    'Data_20140401_03_003',
    'Data_20140401_03_004',
    'Data_20140401_03_025',
    'Data_20140401_03_026',
    'Data_20140401_03_027',
    'Data_20140401_03_028',
    'Data_20140401_03_029',
    'Data_20140401_03_030',
    'Data_20140401_03_031',
    'Data_20140401_03_032',
    'Data_20140401_03_033',
    'Data_20140401_03_034',
    'Data_20140401_03_035',
    'Data_20140401_03_036',
    'Data_20140401_03_037',
    'Data_20140401_03_038',
    'Data_20140401_03_039',
    'Data_20140401_03_040',
    'Data_20140401_03_041',
    'Data_20140401_03_042',
    'Data_20140401_03_043',
    'Data_20140401_03_044',
    'Data_20140401_03_045',
    'Data_20140401_03_046',
    'Data_20140401_03_047',
    #  'Data_20140401_03_048',
    'Data_20140506_01_001',
    'Data_20140506_01_002',
    'Data_20140506_01_003',
    'Data_20140506_01_004',
    'Data_20140506_01_005',
    'Data_20140506_01_006',
    'Data_20140506_01_007',
    'Data_20140506_01_008',
    'Data_20140506_01_009',
    'Data_20140506_01_010',
    'Data_20140506_01_031',
    'Data_20140506_01_032',
    'Data_20140506_01_033',
    'Data_20140506_01_034',
    'Data_20140506_01_035',
    'Data_20140506_01_036',
    'Data_20140506_01_037',
    'Data_20140506_01_038',
    'Data_20140506_01_039',
    'Data_20140506_01_040',
    'Data_20140506_01_041',
    'Data_20140506_01_042',
    'Data_20140506_01_043',
    'Data_20140506_01_044',
    'Data_20140506_01_045',
    #  'Data_20140506_01_046',
]

data_info['test_session_set'] = [
    'Data_20140401_03_005',
    'Data_20140401_03_006',
    'Data_20140401_03_007',
    'Data_20140401_03_008',
    'Data_20140401_03_009',
    'Data_20140401_03_010',
    'Data_20140401_03_011',
    'Data_20140401_03_012',
    'Data_20140401_03_013',
    'Data_20140401_03_014',
    'Data_20140401_03_015',
    'Data_20140401_03_016',
    'Data_20140401_03_017',
    'Data_20140401_03_018',
    'Data_20140401_03_019',
    'Data_20140401_03_020',
    'Data_20140401_03_021',
    'Data_20140401_03_022',
    'Data_20140401_03_023',
    'Data_20140401_03_024',
    'Data_20140506_01_011',
    'Data_20140506_01_012',
    'Data_20140506_01_013',
    'Data_20140506_01_014',
    'Data_20140506_01_015',
    'Data_20140506_01_016',
    'Data_20140506_01_017',
    'Data_20140506_01_018',
    'Data_20140506_01_019',
    'Data_20140506_01_020',
    'Data_20140506_01_021',
    'Data_20140506_01_022',
    'Data_20140506_01_023',
    'Data_20140506_01_024',
    'Data_20140506_01_025',
    'Data_20140506_01_026',
    'Data_20140506_01_027',
    'Data_20140506_01_028',
    'Data_20140506_01_029',
    'Data_20140506_01_030',
]

#Loader and data and rest of the functions taken from CNN_Training
from CNN_Training import CNNloader as loader
from CNN_Training import CNNDataLayer as DataLayer
from CNN_Training import CNNModel as Model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--num_workers', default=4, type=int)

args = parse_args(parser)




os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_sets = {
    phase: DataLayer(
        data_root=args.data_root,
        sessions=getattr(args, phase+'_session_set'),
        loader=loader,
        training=False,
    )
    for phase in args.phases
}

data_loaders = {
    phase: data.DataLoader(
        data_sets[phase],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    for phase in args.phases
}

model = Model().to(device)
model.load_state_dict(torch.load('./snapshots/epoch-20-air-6.0-bed-22.87.pth'))
model.train(False)

with torch.set_grad_enabled(False):
    for phase in args.phases:
        for batch_idx, (data_now, air_target, bed_target, save_path) in enumerate(data_loaders[phase]):
            print('{} {:3.3f}%'.format(phase, 100.0*batch_idx/len(data_loaders[phase])))
#            print('Save path is ')
#            print("Length of save path is ")
#            print(len(save_path))
#            print("first save path")
#            print(save_path[0])
            #print(save_path)

            batch_size = data_now.shape[0]
            data_now = data_now.to(device)
            air_feature, bed_feature = model.features(data_now)
            air_feature = air_feature.to('cpu').numpy()
            bed_feature = bed_feature.to('cpu').numpy()
            for bs in range(batch_size):
                if not osp.isdir(osp.dirname(save_path[bs])):
                    os.makedirs(osp.dirname(save_path[bs]))
                np.save(
                    save_path[bs],
                    np.concatenate((air_feature[bs], bed_feature[bs]), axis=0)
                )
            
            