import os
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
# from utils import *
# from imageio import imread
import h5py as h5

uint16 = 65535


def data_augmentation(x, method):
    if method == 0:
        return np.rot90(x)
    if method == 1:
        return np.fliplr(x)
    if method == 2:
        return np.flipud(x)
    if method == 3:
        return np.rot90(np.rot90(x))
    if method == 4:
        return np.rot90(np.fliplr(x))
    if method == 5:
        return np.rot90(np.flipud(x))


def pre_process_train16(img):
    # img = img // 256
    img = img.astype(np.float32)
    img = img / 65535
    return img


def LDR2HDR(LDR, expo):
    return (LDR ** 2.2) / expo


def imread_uint16_png(image_path, alignratio_path):
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio  # 65535.0


def data_augmentation(x, method):
    if method == 0:
        return np.rot90(x, 2)
    if method == 1:
        return np.fliplr(x)
    if method == 2:
        return np.flipud(x)
    if method == 3:
        return np.rot90(np.rot90(x))
    if method == 4:
        return np.rot90(np.fliplr(x), 2)
    if method == 5:
        return np.rot90(np.flipud(x), 2)


def data_augmentation1(x, method):
    if method == 0:
        return np.rot90(x)
    if method == 1:
        return np.fliplr(x)
    if method == 2:
        return np.flipud(x)
    if method == 3:
        return np.rot90(np.rot90(x))
    if method == 4:
        return np.rot90(np.fliplr(x))
    if method == 5:
        return np.rot90(np.flipud(x))





class Kalantria_train(Dataset):
    def __init__(self, path):
        self.path = path
        file_sub_list = os.listdir(self.path)
        file_sub_list.sort()
        self.num = 0
        self.file_list = []
        for i in range(len(file_sub_list)):
            new_path = path + file_sub_list[i]

            longg_path = new_path + '/' + "1.tif"
            med_path = new_path + '/' + "2.tif"
            short_path = new_path + '/' + "3.tif"
            hdr_path = new_path + '/' + "HDRImg.hdr"
            expos_path = new_path + "/exposure.npy"
            self.file_list += [[longg_path, med_path, short_path, hdr_path, expos_path]]
            self.num += 1

    def __getitem__(self, item):
        #        x_offest,y_offest = random.randint(0,128),random.randint(0,128)
        longg_path, med_path, short_path, hdr_path, expos_path = self.file_list[item]

        longg = cv2.cvtColor(cv2.imread(longg_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0,
                                                                                                        1) / uint16
        med = cv2.cvtColor(cv2.imread(med_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / uint16
        short = cv2.cvtColor(cv2.imread(short_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0,
                                                                                                        1) / uint16
        hdr = cv2.cvtColor(cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        expos = np.load(expos_path).astype(float)
        #        print("expos = ",expos)
        methed = random.randint(0, 5)
        longg = data_augmentation(longg, methed)
        med = data_augmentation(med, methed)
        short = data_augmentation(short, methed)
        hdr = data_augmentation(hdr, methed)
        #        print("end == ",longg.shape,methed)
        longg = np.concatenate([longg, LDR2HDR(longg, (2 ** expos[0]) / (2 ** expos[1]))], axis=0)
        med = np.concatenate([med, LDR2HDR(med, 1)], axis=0)
        short = np.concatenate([short, LDR2HDR(short, (2 ** expos[2]) / (2 ** expos[1]))], axis=0)
        # print("start = ",type(longg))
        #        print("after=",longg.max(),med.max(),short.max())
        # print("end = =",type(longg),methed)
        img4 = (np.log(1 + 5000 * hdr)) / np.log(1 + 5000)
        # print(img1.shape,img2.shape,img3.shape,img4.shape)
        sample = {'input1': longg, 'input2': med, 'input3': short, 'label': img4}
        return sample

    def __len__(self):
        return self.num


class Kalantria_test(Dataset):
    def __init__(self, path):
        self.path = path
        file_sub_list = os.listdir(self.path)
        file_sub_list.sort()
        self.num = 0
        self.file_list = []
        for i in range(len(file_sub_list)):
            new_path = path + file_sub_list[i]

            longg_path = new_path + '/' + "1.tif"
            med_path = new_path + '/' + "2.tif"
            short_path = new_path + '/' + "3.tif"
            hdr_path = new_path + '/' + "HDRImg.hdr"
            expos_path = new_path + "/exposure.txt"
            self.file_list += [[longg_path, med_path, short_path, hdr_path, expos_path]]
            self.num += 1

    def __getitem__(self, item):
        longg_path, med_path, short_path, hdr_path, expos_path = self.file_list[item]

        longg = cv2.cvtColor(cv2.imread(longg_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0,

                                                                                                          1) / uint16
        med = cv2.cvtColor(cv2.imread(med_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / uint16
        short = cv2.cvtColor(cv2.imread(short_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0,
                                                                                                        1) / uint16
        hdr = cv2.cvtColor(cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        expos = np.loadtxt(expos_path).astype(float)
        longg = np.concatenate([longg, LDR2HDR(longg, (2 ** expos[0]) / (2 ** expos[1]))], axis=0)
        med = np.concatenate([med, LDR2HDR(med, 1)], axis=0)
        # short = np.concatenate([short, LDR2HDR(short, (2 ** expos[2]) / (2 ** expos[1]))], axis=0)
        # longg = longg[:, :960:, :1488]
        # med = med[:, :960:, :1488]
        # short = short[:, :960:, :1488]
        # hdr = hdr[:, :960:, :1488]
#        img4 = (np.log(1 + 5000 * hdr)) / np.log(1 + 5000)
        # print(longg.shape,med.shape,short.shape,img4.shape)
        sample = {'input1': longg, 'input2': med, 'input3': short, 'label': img4}
        return sample

    def __len__(self):
        return self.num


class Dataset_h5(Dataset):
    def __init__(self, path):
        super(Dataset_h5, self).__init__()
        self.path = path
        self.file_h5_lists = os.listdir(self.path)
        self.num = len(self.file_h5_lists)

    def __getitem__(self, item):
        single_file = self.file_h5_lists[item]
        h5_file = h5.File(self.path + single_file)
        _x = random.randint(0, 320 - 256)
        _y = random.randint(0, 320 - 256)
        _ldr = h5_file['ldr'][_y:_y + 256, _x:_x + 256][()]
        _hdr = h5_file['hdr'][_y:_y + 256, _x:_x + 256][()]
        #        print("befor=",_ldr.max())
        expos = h5_file['expos']
        #        print("expos = ",expos[()])
        method = random.randint(0, 5)
        _ldr = data_augmentation1(_ldr, method)
        _hdr = data_augmentation1(_hdr, method)
        _ldr = pre_process_train16(_ldr)
        #        print("after=",_ldr.max())
        _x1 = _ldr[:, :, 0:3]
        _x2 = _ldr[:, :, 3:6]
        _x3 = _ldr[:, :, 6:9]
        _x1, _x2, _x3 = _x1.transpose(2, 0, 1), _x2.transpose(2, 0, 1), _x3.transpose(2, 0, 1)
        _hdr = _hdr.transpose(2, 0, 1)
        _hdr = (np.log(1 + 5000 * _hdr)) / np.log(1 + 5000)
        _x1 = np.concatenate([_x1, LDR2HDR(_x1, expos[0] / expos[1])], axis=0)
        _x2 = np.concatenate([_x2, LDR2HDR(_x2, 1)], axis=0)
        _x3 = np.concatenate([_x3, LDR2HDR(_x3, (expos[2]) / (expos[1]))], axis=0)
        #        print("after=",_x1.max(),_x2.max(),_x3.max())
        sample = {'input1': _x1.copy(), 'input2': _x2.copy(), 'input3': _x3.copy(), 'label': _hdr.copy()}

        return sample

    def __len__(self):
        return self.num


class Dataset_h5_test(Dataset):
    def __init__(self, path):
        super(Dataset_h5_test, self).__init__()
        self.path = path
        self.file_h5_lists = os.listdir(self.path)
        self.num = len(self.file_h5_lists)

    def __getitem__(self, item):
        single_file = self.file_h5_lists[item]
        h5_file = h5.File(self.path + single_file)

        _ldr = h5_file['ldr'][:992, :1488, :][()]
        _hdr = h5_file['hdr'][:992, :1488, :][()]

        expos = h5_file['expos']
        _ldr = pre_process_train16(_ldr)

        _x1 = _ldr[:, :, 0:3]
        _x2 = _ldr[:, :, 3:6]
        _x3 = _ldr[:, :, 6:9]
        _x1, _x2, _x3 = _x1.transpose(2, 0, 1), _x2.transpose(2, 0, 1), _x3.transpose(2, 0, 1)
        _hdr = _hdr.transpose(2, 0, 1)
#        _hdr = (np.log(1 + 5000 * _hdr)) / np.log(1 + 5000)
        _x1 = np.concatenate([_x1, LDR2HDR(_x1, expos[0] / expos[1])], axis=0)
        _x2 = np.concatenate([_x2, LDR2HDR(_x2, 1)], axis=0)
        _x3 = np.concatenate([_x3, LDR2HDR(_x3, (expos[2]) / (expos[1]))], axis=0)
        sample = {'input1': _x1.copy(), 'input2': _x2.copy(), 'input3': _x3.copy(), 'label': _hdr.copy()}

        return sample

    def __len__(self):
        return self.num


class MyDataset_train(Dataset):
    def __init__(self, data_path, size):
        file_list = os.listdir(data_path)
        file_list.sort()
        self.size = size
        self.data_list = []
        self.files_list = []
        self.num = 0
        self.all_length = 0
        self.offest_x = 36
        self.offest_y = 108
        self.x = 1060
        self.y = 1900

        for j in range(len(file_list) - 6):
            # if j>0:
            #    self.all_length += (len(os.listdir(data_path + file_list[j - 1])) // 6)
            new_path = os.path.join(data_path, file_list[j])
            start = int(file_list[j][5:9])
            end = int(file_list[j][-4:])
            for tmp in range(start, end + 1):
                # print(len(os.listdir(new_path)))
                # tmp = self.all_length+i
                # print(tmp)
                if tmp < 10:
                    tmp = '000' + str(tmp)
                elif tmp < 100:
                    tmp = '00' + str(tmp)
                elif tmp < 1000:
                    tmp = '0' + str(tmp)
                else:
                    tmp = str(tmp)
                expos_name = new_path + '/' + tmp + '_exposures.npy'
                longg_name = new_path + '/' + tmp + '_long.png'
                med_name = new_path + '/' + tmp + '_medium.png'
                short_name = new_path + '/' + tmp + '_short.png'
                hdr_name = new_path + '/' + tmp + '_gt.png'
                align_name = new_path + '/' + tmp + '_alignratio.npy'
                self.files_list += [[expos_name, longg_name, med_name, short_name, hdr_name, align_name]]
                self.num = len(self.files_list) * (self.y // self.size) * (self.x // self.size)

    def __getitem__(self, item):
        loc = item % ((self.y // self.size) * (self.x // self.size))
        index = item // ((self.y // self.size) * (self.x // self.size))
        # print(loc,index)
        expos_name, longg_name, med_name, short_name, hdr_name, align_name = self.files_list[index]

        expos = np.load(expos_name).astype(float)
        longg = cv2.cvtColor(cv2.imread(longg_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        med = cv2.cvtColor(cv2.imread(med_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        short = cv2.cvtColor(cv2.imread(short_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        longg = longg / 255
        med = med / 255
        short = short / 255

        hdr = imread_uint16_png(hdr_name, align_name)
        hdr = hdr.transpose(2, 0, 1)
        longg = np.concatenate([longg, LDR2HDR(longg, (2 ** expos[0]) / (2 ** expos[1]))], axis=0)
        med = np.concatenate([med, LDR2HDR(med, 1)], axis=0)
        short = np.concatenate([short, LDR2HDR(short, (2 ** expos[2]) / (2 ** expos[1]))], axis=0)
        x_len = self.x // self.size

        _x, _y = (loc % x_len), (loc // x_len)
        # print(_x,_y)
        offset_x, offset_y = random.randint(0, self.offest_x - 1), random.randint(0, self.offest_y - 1)
        img1 = longg[:, _x * self.size + offset_x:(_x + 1) * self.size + offset_x,
               _y * self.size + offset_y:(_y + 1) * self.size + offset_y]
        img2 = med[:, _x * self.size + offset_x:(_x + 1) * self.size + offset_x,
               _y * self.size + offset_y:(_y + 1) * self.size + offset_y]
        img3 = short[:, _x * self.size + offset_x:(_x + 1) * self.size + offset_x,
               _y * self.size + offset_y:(_y + 1) * self.size + offset_y]
        img4 = hdr[:, _x * self.size + offset_x:(_x + 1) * self.size + offset_x,
               _y * self.size + offset_y:(_y + 1) * self.size + offset_y]
        img4 = (np.log(1 + 5000 * img4)) / np.log(1 + 5000)
        # print(img1.shape,img2.shape,img3.shape,img4.shape)
        sample = {'input1': img1, 'input2': img2, 'input3': img3, 'label': img4}
        return sample

    def __len__(self):
        return self.num


class MyDataset_128_train(Dataset):
    def __init__(self, data_path):
        file_list = os.listdir(data_path)
        file_list.sort()
        self.data_list = []
        self.files_list = []
        self.num = 0
        self.all_length = 0
        for index in range(0, len(file_list), 6):
            tmp = int(file_list[index][:4])
            if tmp < 10:
                tmp = '000' + str(tmp)
            elif tmp < 100:
                tmp = '00' + str(tmp)
            elif tmp < 1000:
                tmp = '0' + str(tmp)
            else:
                tmp = str(tmp)
            sec = int(file_list[index][6:9])
            if sec < 10:
                sec = '000' + str(sec)
            elif sec < 100:
                sec = '00' + str(sec)
            elif sec < 1000:
                sec = '0' + str(sec)
            else:
                sec = str(sec)
            expos_name = data_path + '/' + tmp + "_" + sec + '_exposures.npy'
            align_name = data_path + '/' + tmp + "_" + sec + '_alignratio.npy'
            longg_name = data_path + '/' + tmp + "_" + sec + '_long.png'
            med_name = data_path + '/' + tmp + "_" + sec + '_medium.png'
            short_name = data_path + '/' + tmp + "_" + sec + '_short.png'
            hdr_name = data_path + '/' + tmp + "_" + sec + '_gt.png'
            self.files_list += [[expos_name, longg_name, med_name, short_name, hdr_name, align_name]]
            self.num += 1

    def __getitem__(self, item):
        expos_name, longg_name, med_name, short_name, hdr_name, align_name = self.files_list[item]

        expos = np.load(expos_name).astype(float)
        longg = cv2.cvtColor(cv2.imread(longg_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        med = cv2.cvtColor(cv2.imread(med_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        short = cv2.cvtColor(cv2.imread(short_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        longg = longg / 255.0
        med = med / 255.0
        short = short / 255.0

        hdr = imread_uint16_png(hdr_name, align_name)
        hdr = hdr.transpose(2, 0, 1)
        longg = np.concatenate([longg, LDR2HDR(longg, (2 ** expos[0]) / (2 ** expos[1]))], axis=0)
        med = np.concatenate([med, LDR2HDR(med, 1)], axis=0)
        short = np.concatenate([short, LDR2HDR(short, (2 ** expos[2]) / (2 ** expos[1]))], axis=0)

        img4 = (np.log(1 + 5000 * hdr)) / np.log(1 + 5000)
        # print(img1.shape,img2.shape,img3.shape,img4.shape)
        sample = {'input1': longg, 'input2': med, 'input3': short, 'label': img4}
        return sample

    def __len__(self):
        return self.num


class MyDataset_test(Dataset):
    def __init__(self, data_path):
        file_list = os.listdir(data_path)
        file_list.sort()
        self.data_list = []
        self.files_list = []
        self.num = 0
        self.all_length = 0

        for j in range(len(file_list) - 1, len(file_list)):
            # if j>0:
            #    self.all_length += (len(os.listdir(data_path + file_list[j - 1])) // 6)
            new_path = os.path.join(data_path, file_list[j])
            start = int(file_list[j][5:9])
            end = int(file_list[j][-4:])
            for tmp in range(start, end + 1):
                # print(len(os.listdir(new_path)))
                # tmp = self.all_length+i
                # print(tmp)
                if tmp < 10:
                    tmp = '000' + str(tmp)
                elif tmp < 100:
                    tmp = '00' + str(tmp)
                elif tmp < 1000:
                    tmp = '0' + str(tmp)
                else:
                    tmp = str(tmp)
                expos_name = new_path + '/' + tmp + '_exposures.npy'
                longg_name = new_path + '/' + tmp + '_long.png'
                med_name = new_path + '/' + tmp + '_medium.png'
                short_name = new_path + '/' + tmp + '_short.png'
                hdr_name = new_path + '/' + tmp + '_gt.png'
                align_name = new_path + '/' + tmp + '_alignratio.npy'
                self.files_list += [[expos_name, longg_name, med_name, short_name, hdr_name, align_name]]
                self.num = len(self.files_list)  # hdr.shape[0]//256, hdr.shape[1]//256

    def __getitem__(self, item):
        expos_name, longg_name, med_name, short_name, hdr_name, align_name = self.files_list[item]

        expos = np.load(expos_name).astype(float)
        longg = cv2.cvtColor(cv2.imread(longg_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        med = cv2.cvtColor(cv2.imread(med_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        short = cv2.cvtColor(cv2.imread(short_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        longg = longg / 255
        med = med / 255
        short = short / 255

        hdr = imread_uint16_png(hdr_name, align_name)
        hdr = hdr.transpose(2, 0, 1)
        longg = np.concatenate([longg, LDR2HDR(longg, (2 ** expos[0]) / (2 ** expos[1]))], axis=0)
        med = np.concatenate([med, LDR2HDR(med, 1)], axis=0)
        short = np.concatenate([short, LDR2HDR(short, (2 ** expos[2]) / (2 ** expos[1]))], axis=0)
        img1 = longg[:, :1024, :1024]
        img2 = med[:, :1024, :1024]
        img3 = short[:, :1024, :1024]
        img4 = hdr[:, :1024, :1024]
        img4 = (np.log(1 + 5000 * img4)) / np.log(1 + 5000)
        # print(img1.shape,img2.shape,img3.shape,img4.shape)
        sample = {'input1': img1, 'input2': img2, 'input3': img3, 'label': img4}
        return sample

    def __len__(self):
        return self.num
