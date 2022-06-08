from os.path import join
from os import listdir
from scipy.io import loadmat

from torch.utils import data
import numpy as np

# from utils.augmentation_cpu import MirrorTransform, SpatialTransform


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])

def imgnorm(N_I, index1=0.05, index2=0.05):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, file_dir, num_classes, is_rand_anti=True):
        super(DatasetFromFolder3D, self).__init__()
        self.image_filenames = [x for x in listdir(file_dir) if is_image_file(x)]
        self.file_dir = file_dir
        self.num_classes = num_classes
        self.is_rand_anti = is_rand_anti

    def __getitem__(self, index):
        # 读取image和label
        data = loadmat(join(self.file_dir, self.image_filenames[index]))

        fix_img = data['fix_img']
        fix_img = np.where(fix_img < 0., 0., fix_img)
        fix_img = np.where(fix_img > 2048., 2048., fix_img)
        fix_img = fix_img / 2048.
        fix_img = fix_img.astype(np.float32)

        mov_img = data['mov_img']
        mov_img = np.where(mov_img < 0., 0., mov_img)
        mov_img = np.where(mov_img > 2048., 2048., mov_img)
        mov_img = mov_img / 2048.
        mov_img = mov_img.astype(np.float32)

        if len(data) == 6:
            mov_lab = data['mov_lab']
            mov_lab = np.where(mov_lab == 205, 1, mov_lab)
            mov_lab = np.where(mov_lab == 420, 2, mov_lab)
            mov_lab = np.where(mov_lab == 500, 3, mov_lab)
            mov_lab = np.where(mov_lab == 550, 4, mov_lab)
            mov_lab = np.where(mov_lab == 600, 5, mov_lab)
            mov_lab = np.where(mov_lab == 820, 6, mov_lab)
            mov_lab = np.where(mov_lab == 850, 7, mov_lab)
            fix_lab = 0
        else:
            mov_lab = 0
            fix_lab = 0

        if len(data) == 6:
            mov_lab = self.to_categorical(mov_lab, self.num_classes)
            mov_lab = mov_lab.astype(np.float32)

        mov_img = mov_img[np.newaxis, :, :, :]
        fix_img = fix_img[np.newaxis, :, :, :]

        # 是否交换
        if self.is_rand_anti:
            is_anti = np.random.randint(low=0, high=2)
            if is_anti == 1:
                t = mov_img
                mov_img = fix_img
                fix_img = t
                fix_lab = mov_lab
                mov_lab = 0
        return mov_img, fix_img, mov_lab, fix_lab

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.image_filenames)


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical


if __name__ == '__main__':
    # 读取image和label
    data = loadmat('/media/E/yt/RST/data/heart_CT_5/train_labeled_labeled/1001_1002.mat')
    print(len(data))

    fix_img = data['fix_img']
    fix_img = np.where(fix_img < 0., 0., fix_img)
    fix_img = np.where(fix_img > 2048., 2048., fix_img)
    fix_img = fix_img / 2048.
    fix_img = fix_img.astype(np.float32)

    mov_img = data['mov_img']
    mov_img = np.where(mov_img < 0., 0., mov_img)
    mov_img = np.where(mov_img > 2048., 2048., mov_img)
    mov_img = mov_img / 2048.
    mov_img = mov_img.astype(np.float32)

    mov_lab = data['mov_lab']
    print(mov_lab.shape)

    mov_lab = data['mov_lab']
    mov_lab = np.where(mov_lab == 205, 1, mov_lab)
    mov_lab = np.where(mov_lab == 420, 2, mov_lab)
    mov_lab = np.where(mov_lab == 500, 3, mov_lab)
    mov_lab = np.where(mov_lab == 550, 4, mov_lab)
    mov_lab = np.where(mov_lab == 600, 5, mov_lab)
    mov_lab = np.where(mov_lab == 820, 6, mov_lab)
    mov_lab = np.where(mov_lab == 850, 7, mov_lab)
    print(mov_lab.shape)

    x = to_categorical(mov_lab)
    print(x.shape)
    print(np.max(x))