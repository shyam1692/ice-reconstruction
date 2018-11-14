#!/usr/bin/env python3
"""
Created on Thu Nov  1 20:38:07 2018

@author: Shyam
Writing code for RNN framework

"""
import os
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim
import torch.utils.data as data

import os.path as osp
import numpy.linalg as LA
def fc_relu(in_features, out_features, inplace=True):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(inplace=inplace),
        nn.Dropout(p=0.1),
    )

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.hsize = 512

        self.air_rnn_0 = nn.GRUCell(self.hsize, self.hsize)
        self.bed_rnn_0 = nn.GRUCell(self.hsize, self.hsize)
        self.air_rnn_1 = nn.GRUCell(self.hsize, self.hsize)
        self.bed_rnn_1 = nn.GRUCell(self.hsize, self.hsize)

        self.air_fc_in_0 = fc_relu(64, self.hsize)
        self.bed_fc_in_0 = fc_relu(64, self.hsize)
        self.air_fc_in_1 = fc_relu(64, self.hsize)
        self.bed_fc_in_1 = fc_relu(64, self.hsize)
        self.air_fc_out = nn.Linear(self.hsize, 1)
        self.bed_fc_out = nn.Linear(self.hsize, 1)

    def forward(self, data, init):
        air_output = None
        bed_output = None
        air_hidden = [[data.new_zeros((data.shape[0], self.hsize)) for i in range(65)] for j in range(2)]
        bed_hidden = [[data.new_zeros((data.shape[0], self.hsize)) for i in range(65)] for j in range(2)]
        air_hidden[0][0] = init
        bed_hidden[0][0] = init
        air_hidden[1][0] = init
        bed_hidden[1][0] = init

        for i in range(64):
            air_input_0 = self.air_fc_in_0(data[:,:,i])
            bed_input_0 = self.bed_fc_in_0(data[:,:,i])
            air_input_1 = self.air_fc_in_1(data[:,:,63-i])
            bed_input_1 = self.bed_fc_in_1(data[:,:,63-i])

            air_hidden[0][i+1] = self.air_rnn_0(air_input_0, air_hidden[0][i])
            bed_hidden[0][i+1] = self.bed_rnn_0(bed_input_0, bed_hidden[0][i])
            air_hidden[1][i+1] = self.air_rnn_1(air_input_1, air_hidden[1][i])
            bed_hidden[1][i+1] = self.bed_rnn_1(bed_input_1, bed_hidden[1][i])

        for i in range(1, 65):
            air_temp = self.air_fc_out(air_hidden[0][i]+air_hidden[1][65-i])
            bed_temp = self.bed_fc_out(bed_hidden[0][i]+bed_hidden[1][65-i])

            air_output = air_temp if i ==1 else torch.cat((air_output, air_temp), 1)
            bed_output = bed_temp if i ==1 else torch.cat((bed_output, bed_temp), 1)

        return air_output, bed_output
    
    
#Now for RNN Dataloader, there doesn't seem to be one
#Let's check RNN datalayer
class RNNDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, features='c2d_features'):
        self.data_root = data_root
        self.sessions = sessions
        self.features = features

        self.inputs = []
        for session_name in self.sessions:
            session_path = osp.join(self.data_root, 'target', session_name+'.txt')
            session_data = open(session_path, 'r').read().splitlines()
            self.inputs.extend(session_data)

    def rnn_loader(self, path, number):
        data_path = osp.join(self.data_root, 'slices_npy_64x64', path)
        data = np.load(osp.join(data_path, number.zfill(5)+'.npy'))
        data = data = (data-0.5)/0.5
        norm = LA.norm(data, axis=0)
        data /= norm[None, :]
        init_path = osp.join(self.data_root, self.features, path)
        init = np.load(osp.join(init_path, number.zfill(5)+'.npy'))
        return data, init

    def __getitem__(self, index):
        path, number, air_target, bed_target = self.inputs[index].split()
        data, init = self.rnn_loader(path, number)
        data = torch.from_numpy(data)
        init = torch.from_numpy(init)
        air_target = np.array(air_target.split(','), dtype=np.float32)
        air_target = torch.from_numpy(air_target)
        bed_target = np.array(bed_target.split(','), dtype=np.float32)
        bed_target = torch.from_numpy(bed_target)
        return data, init, air_target, bed_target

    def __len__(self):
        return len(self.inputs)

#Next is training
def weights_init_rnn(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', default='0,1,2,3', type=str)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=1e-03, type=float)
parser.add_argument('--num_workers', default=4, type=int)

from collections import OrderedDict
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


parser.add_argument('--data_root', default='../data', type=str)
parser.add_argument('--phases', default=['train', 'test'], type=list)
parser.add_argument('--train_session_set', default=data_info['train_session_set'], type=list)
parser.add_argument('--test_session_set', default=data_info['test_session_set'], type=list)
parser.add_argument('--test_interval', default=1, type=int)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Device is:')
print(device)

data_sets = {
    phase: RNNDataLayer(
        data_root=args.data_root,
        sessions=getattr(args, phase+'_session_set'),
    )
    for phase in args.phases
}

data_loaders = {
    phase: data.DataLoader(
        data_sets[phase],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    for phase in args.phases
}


model = RNN().apply(weights_init_rnn).to(device)
air_criterion = nn.L1Loss().to(device)
bed_criterion = nn.L1Loss().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

import time
text_to_print = []
for epoch in range(1, args.epochs+1):
    # Learning rate scheduler
    if epoch == 5 or epoch%10 == 0 :
        args.lr = args.lr * 0.4
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    air_errors = {phase: 0.0 for phase in args.phases}
    bed_errors = {phase: 0.0 for phase in args.phases}

    start = time.time()
    for phase in args.phases:
        training = phase=='train'
        if training:
            model.train(True)
        else:
            if epoch%args.test_interval == 0:
                model.train(False)
            else:
                continue

        with torch.set_grad_enabled(training):
            for batch_idx, (data_now, init, air_target, bed_target) in enumerate(data_loaders[phase]):
                print('Epoch is')
                print(epoch)
                print('Batch ID is')
                print(batch_idx)
                batch_size = data_now.shape[0]
                data_now = data_now.to(device)
                init = init.to(device)
                air_target = air_target.to(device)
                bed_target = bed_target.to(device)

                air_output, bed_output = model(data_now, init)
                air_loss = air_criterion(air_output, air_target)
                bed_loss = bed_criterion(bed_output, bed_target)
                air_errors[phase] += air_loss.item()*batch_size
                bed_errors[phase] += bed_loss.item()*batch_size
                if args.debug:
                    print(air_loss.item(), bed_loss.item())

                if training:
                    optimizer.zero_grad()
                    loss = air_loss + bed_loss
                    loss.backward()
                    optimizer.step()
    end = time.time()

    if epoch%args.test_interval == 0:
        snapshot_path = './snapshots_rnn'
        if not os.path.isdir(snapshot_path):
            os.makedirs(snapshot_path)
        snapshot_name = 'epoch-{}-air-{}-bed-{}.pth'.format(
            epoch,
            float("{:.2f}".format(air_errors['test']/len(data_loaders['test'].dataset)*412)),
            float("{:.2f}".format(bed_errors['test']/len(data_loaders['test'].dataset)*412)),
        )
        torch.save(model.state_dict(), os.path.join(snapshot_path, snapshot_name))

    text_to_append = ('Epoch {:2}, | '
          'train loss (air): {:4.2f} (bed): {:4.2f}, | '
          'test loss (air): {:4.2f} (bed): {:4.2f}, | '
          'running time: {:.2f} sec'.format(
              epoch,
              air_errors['train']/len(data_loaders['train'].dataset)*412,
              bed_errors['train']/len(data_loaders['train'].dataset)*412,
              air_errors['test']/len(data_loaders['test'].dataset)*412,
              bed_errors['test']/len(data_loaders['test'].dataset)*412,
              end-start,
          ))
    text_to_print.append(text_to_append)
    
for text in text_to_print:
    print(text)
