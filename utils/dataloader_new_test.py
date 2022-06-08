from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii.gz"]) and int(filename[:3]) > 90


def imgnorm(N_I, index1=0.05, index2=0.05):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min+1e-6)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


def limit(image):
    max = np.where(image < 0)
    image[max] = 0
    return image


def Nor(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data


class DatasetFromFolder3D_Test(data.Dataset):
    def __init__(self, file_dir, num_classes, is_rand_anti=True):
        super(DatasetFromFolder3D_Test, self).__init__()
        self.pairs = [i.strip() for i in open(join(file_dir, 'test.txt'), 'r').readlines()]
        self.file_dir = file_dir
        self.num_classes = num_classes
        self.list = [0,2,3,4,7,8,10,11,12,13,14,15,16,17,18,24,28,41,42,43,46,47,49,50,51,52,53,54,60]

    def __getitem__(self, index):
        mov_index, fix_index = self.pairs[index].split('_')[0], self.pairs[index].split('_')[1]

        mov_img, mov_lab = self.get_pair(mov_index)
        fix_img, fix_lab = self.get_pair(fix_index)

        name = self.pairs[index]

        return mov_img, fix_img, mov_lab, fix_lab, name

    def get_pair(self, index):
        img = sitk.ReadImage(join(self.file_dir, 'image', index + '.nii.gz'))
        img = sitk.GetArrayFromImage(img)
        img = Nor(limit(img))
        img = img.astype(np.float32)

        lab = sitk.ReadImage(join(self.file_dir, 'label', index + '.nii.gz'))
        lab = sitk.GetArrayFromImage(lab)
        mask = np.where(lab > 0, 1, 0).astype(np.float32)
        img = img * mask
        img = img[np.newaxis, :, :, :]

        lab = [np.where(lab == i, 1, 0)[np.newaxis, :, :, :] for i in self.list]
        lab = np.concatenate(lab, axis=0)
        lab = lab.astype(np.float32)

        return img, lab

    def __len__(self):
        return len(self.pairs)