#!/usr/bin/env python3
"""
Created on Thu Oct 25 18:25:52 2018

@author: Shyam

Following is the code for just CNN training, where we will be saving the weights, creating checkpoints,
and training in the GPU across multiple workers.
"""
import os
import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1_s = nn.Sequential(
            nn.Conv2d(1, 16, (5, 3), stride=1, padding=(2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(( 2, 2)),
        )

        self.conv2_s = nn.Sequential(
            nn.Conv2d(16, 32, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(( 2, 2)),
        )

        self.conv3_a = nn.Sequential(
            nn.Conv2d(32, 64, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(( 2, 2)),
        )

        self.conv3_b = nn.Sequential(
            nn.Conv2d(32, 64, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (5, 3), stride=1, padding=(2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )

        self.conv4_a = nn.Sequential(
            nn.Conv2d(64, 128, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(( 2, 2)),
        )

        self.conv4_b = nn.Sequential(
            nn.Conv2d(64, 128, (5, 3), stride=1, padding=(2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, ( 5, 3), stride=1, padding=( 2, 1)),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(( 2, 2)),
        )

        self.conv5_a = nn.Sequential(
            nn.Conv2d(128, 256, ( 4, 4), stride=1, padding=( 0, 0)),
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        self.conv5_b = nn.Sequential(
            nn.Conv2d(128, 256, ( 4, 4), stride=1, padding=( 0, 0)),
            #nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            Flatten(),
        )

        self.fc6_a = nn.Linear(256, 64)
        self.fc6_b = nn.Linear(256, 64)

    def features(self, img):
        img = self.conv1_s(img)
        img = self.conv2_s(img)
        air = self.conv3_a(img)
        bed = self.conv3_b(img)
        air = self.conv4_a(air)
        bed = self.conv4_b(bed)
        air = self.conv5_a(air)
        bed = self.conv5_b(bed)
        return air, bed

    def forward(self, img):
        air, bed = self.features(img)
        air = self.fc6_a(air)
        bed = self.fc6_b(bed)
        return air, bed





#Training the data
import torch.optim as optim
import torch.utils.data as data

#In loader, we are taking 5 consecutive images and making dimension as 5*64*64
#Then we increase the dimension to make it as 1*5*64*64
#We are increasing the dimension to signify input channel.
#Let's say that we are at index 7. We take images from Index &-2 = 5 to 7 + 2 =9.(5 to 9)
#In CNNDataLayer function, we will see where the output is taken from.
def CNNloader(path, number):
    data = np.load(osp.join(path, str(number).zfill(5)+'.npy'))
    data = np.array(data, dtype=np.float32)
    data = (data-0.5)/0.5
    data = data[np.newaxis, ...]
    return data



import os.path as osp
#Basically, each file, let's say with number n, has image of 64*64
#The same file has target associated with it.
#Example, we have the file as '20140325_05/001/21' as file name, where n = 21.
#Then what we do is, the target variable y will be target variable of '20140325_05/001/21' (64 dimension)
#But the 3D input will be '20140325_05/001/19' to '20140325_05/001/23' (n-2 to n+2, as seen in the CNNloader function)
class CNNDataLayer(data.Dataset):
    def __init__(self, data_root, sessions, loader, training=True):
        self.data_root = data_root
        self.sessions = sessions
        self.loader = loader
        self.training = training

        self.inputs = []
        for session_name in self.sessions:
            session_path = osp.join(self.data_root, 'target', session_name+'.txt')
            session_data = open(session_path, 'r').read().splitlines()
            self.inputs.extend(session_data)

    def __getitem__(self, index):
        data_path, number, air_target, bed_target = self.inputs[index].split()
        data = self.loader(osp.join(self.data_root, 'slices_npy_64x64', data_path), int(number))
        data = torch.from_numpy(data)
        air_target = np.array(air_target.split(','), dtype=np.float32)
        air_target = torch.from_numpy(air_target)
        bed_target = np.array(bed_target.split(','), dtype=np.float32)
        bed_target = torch.from_numpy(bed_target)

        if self.training:
            return data, air_target, bed_target
        else:
            save_path = osp.join(self.data_root, 'c2d_features', data_path, number.zfill(5)+'.npy')
            return data, air_target, bed_target, save_path

    def __len__(self):
        return len(self.inputs)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1.0, std=0.001)
        nn.init.constant_(m.bias.data, 0.001)
        
if __name__ == "__main__":
    import argparse
    #import config as cfg
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-04, type=float)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    #Following for working in GPU
    #parser.add_argument('--num_workers', default=4, type=int)
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
    
    
    
    #One line not working
    #parser.add_argument('--data_root', default='/l/vision/v7/mx6/data/CReSIS', type=str)
    parser.add_argument('--data_root', default='../data', type=str)
    #C:\stuff\Studies\Fall 18\Independent Study\Week 5\Data
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--train_session_set', default=data_info['train_session_set'], type=list)
    parser.add_argument('--test_session_set', default=data_info['test_session_set'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_sets = {
        phase: CNNDataLayer(
            data_root=args.data_root,
            sessions=getattr(args, phase+'_session_set'),
            loader=CNNloader,
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
    
    print("The device is")
    print(device)
    model = CNNModel().apply(weights_init).to(device)
    print("This time we are not at all doing dropout")
    
    air_criterion = nn.L1Loss().to(device)
    bed_criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    
    text_to_print = []
    import time
    for epoch in range(1, args.epochs+1):
        # Learning rate scheduler
        if epoch%5 == 0 and args.lr >= 1e-05:
            args.lr = args.lr * 0.5
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
                for batch_idx, (data_now, air_target, bed_target) in enumerate(data_loaders[phase]):
                    print('Epoch is')
                    print(epoch)
                    print('Batch ID is')
                    print(batch_idx)
                    batch_size = data_now.shape[0]
                    data_now = data_now.to(device)
                    air_target = air_target.to(device)
                    bed_target = bed_target.to(device)
    
                    air_output, bed_output = model(data_now)
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
            snapshot_path = './snapshots'
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
        
