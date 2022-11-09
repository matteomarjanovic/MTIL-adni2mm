from __future__ import print_function, division
from torch.utils.data import Dataset
from utils import read_csv_mt
import numpy as np
import csv
import random
import scipy.ndimage.interpolation as sni


# rotate volume
def rot_vol(vol, angle = 90, axis = 'z', int_order = 0, reshape=True, cval=0.0):

    if (axis == 'x'):
        a = 0
        b = 2
    if (axis == 'y'):
        a = 1
        b = 2
    if (axis == 'z'):
        a = 0
        b = 1

    vol = np.swapaxes(vol,a,b)
    vol_r = sni.rotate(vol, angle, axes=(1, 0), reshape=reshape, output=None, order=int_order, mode='constant', cval=cval, prefilter=True)
    vol_r = np.swapaxes(vol_r,a,b)

    return vol_r



def aug_vol(inv,normalize=True):

    # rotate
    if random.random()> .5:

        angle_x = random.randrange(-30,30)/10
        #angle_y = random.randrange(-50,50)/10
        angle_z = random.randrange(-50,50)/10

        tmp = rot_vol(inv,angle=angle_x,axis='x',int_order=0,reshape=True)
        rot = rot_vol(tmp,angle=angle_z,axis='z',int_order=0,reshape=True) 
                

        aux_center = np.asarray(rot.shape)/2

        # inv = rot[int(aux_center[0]) - int(91/2):int(aux_center[0]) + int(91/2)+1, int(aux_center[1]) - int(109/2):int(aux_center[1]) + int(109/2)+1, int(aux_center[2]) - int(91/2):int(aux_center[2]) + int(91/2)+1]
        inv = rot[int(aux_center[0]) - int(91):int(aux_center[0]) + int(91), int(aux_center[1]) - int(109):int(aux_center[1]) + int(109), int(aux_center[2]) - int(91):int(aux_center[2]) + int(91)]

        # inv = rot
        #inv = rot[int(aux_center[0]) - 45:int(aux_center[0]) + 45, int(aux_center[1]) - int(125/2):int(aux_center[1]) + int(125/2)+1, int(aux_center[2]) - int(125/2):int(aux_center[2]) + int(125/2)+1]
        #inv = rot[int(aux_center[0]) - 90:int(aux_center[0]) + 90, int(aux_center[1]) - int(240/2):int(aux_center[1]) + int(240/2)+1, int(aux_center[2]) - int(240/2):int(aux_center[2]) + int(240/2)+1]


    # random Gaussian noise
    noise1 = np.random.normal(0, 0.1, size=inv.shape)
    noise1[np.abs(noise1) > 0.1] = 0
    inv = inv + noise1
    del noise1

    # random translation
    if random.random()> .5:
        off_set_x = random.randint(-3,3)#
        off_set_y = random.randint(-3,3)#
        off_set_z = random.randint(-3,3)#
        if off_set_x != 0 and off_set_y !=0 and off_set_z !=0:

            tr = np.zeros(inv.shape)
            if off_set_x < 0:
                tr[0:tr.shape[0]-abs(off_set_x),:,:] = inv[abs(off_set_x):,:,:]
            if off_set_x > 0:
                tr[off_set_x:,:,:] = inv[0:-abs(off_set_x),:,:]

            if off_set_y < 0:
                tr[:,0:tr.shape[1]-abs(off_set_y),:] = inv[:,abs(off_set_y):,:]
            if off_set_y > 0:
                tr[:,off_set_y:,:] = inv[:,0:-abs(off_set_y),:]

            if off_set_z < 0:
                tr[:,:,0:tr.shape[2]-abs(off_set_z)] = inv[:,:,abs(off_set_z):]
            if off_set_z > 0:
                tr[:,:,off_set_z:] = inv[:,:,0:-abs(off_set_z)]

            inv = tr
            del tr

    # flip on x-axes
    if random.random() > .5:
        inv = inv[::-1,:,:]

    return inv    



class CNN_Data(Dataset):
    """
    csv files ./lookuptxt/*.csv contains MRI filenames along with demographic and diagnosis information 
    """
    def __init__(self, Data_dir, stage, dataset, cross_index, start, end, seed=1000):
        # random.seed(seed)
        self.Data_dir = Data_dir
        self.Data_list, self.Label_list, self.demor_list = read_csv_mt('./lookupcsv/{}.csv'.format(dataset))
        train_data_list = list()
        train_label_list = list()
        train_demor_list = list()
        test_data_list = list()
        test_label_list = list()
        test_demor_list = list()
        with open("./lookupcsv/{}.csv".format(dataset), 'r') as file:
            f_reader = csv.reader(file)
            for i, row in enumerate(f_reader):
                if i == 0:
                    continue
                if end == -1 or start <= i - 1 <= end:
                    test_data_list.append(row[0])
                    test_label_list.append(0 if row[1] == "AD" else 1)
                    # test_demor_list.append(int(row[2]))
                    test_demor_list.append(float(row[2]))
                else:
                    train_data_list.append(row[0])
                    train_label_list.append(0 if row[1] == "AD" else 1)
                    # train_demor_list.append(int(row[2]))
                    train_demor_list.append(float(row[2]))
        num = end - start
        if stage == 'valid':
            if cross_index != 9:
                self.Data_list = train_data_list[num * cross_index:num * (cross_index + 1)]
                self.Label_list = train_label_list[num * cross_index:num * (cross_index + 1)]
                self.demor_list = train_demor_list[num * cross_index:num * (cross_index + 1)]
            else:
                self.Data_list = train_data_list[:num]
                self.Label_list = train_label_list[:num]
                self.demor_list = train_demor_list[:num]
        elif stage == 'train':
            if cross_index != 9:
                self.Data_list = train_data_list[:num * cross_index] + train_data_list[num * (cross_index + 1):]
                self.Label_list = train_label_list[:num * cross_index] + train_label_list[num * (cross_index + 1):]
                self.demor_list = train_demor_list[:num * cross_index] + train_demor_list[num * (cross_index + 1):]
            else:
                self.Data_list = train_data_list[num:]
                self.Label_list = train_label_list[num:]
                self.demor_list = train_demor_list[num:]
        else:
            self.Data_list = test_data_list
            self.Label_list = test_label_list
            self.demor_list = test_demor_list

    def __len__(self):
        return len(self.Data_list)

    def __getitem__(self, idx):
        label = self.Label_list[idx]
        demor = self.demor_list[idx]
        data = np.load(self.Data_dir + self.Data_list[idx] + ".npy").astype(np.float32)
        # data = aug_vol(data).copy()
        data = np.expand_dims(data, axis=0)
        # return data.astype(np.float32), label, np.asarray(demor).astype(np.float32)
        return data, label, np.asarray(demor).astype(np.float32)

    def get_sample_weights(self):
        count, count0, count1 = float(len(self.Label_list)), float(self.Label_list.count(0)), float(self.Label_list.count(1))
        weights = [count / count0 if i == 0 else count / count1 for i in self.Label_list]
        return weights, count0 / count1
