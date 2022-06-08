import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.XMorpher import Head
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import MirrorTransform, SpatialTransform
from utils.dataloader import DatasetFromFolder3D
from utils.dataloader_test import DatasetFromFolder3D_Test
from utils.losses import gradient_loss, ncc_loss, crossentropy, MSE, mask_crossentropy
from utils.utils import EMA, AverageMeter, to_categorical, dice
import numpy as np

import time


class RSeg(object):
    def __init__(self, k=0, n_channels=1, n_classes=8, lr=1e-4, epoches=400, iters=200, batch_size=1, model_name='XMorpher_heart_0'):
        super(RSeg, self).__init__()
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters=iters
        train_labeled_unlabeled_dir = 'data/train_labeled_unlabeled'
        train_unlabeled_unlabeled_dir = 'data/train_unlabeled_unlabeled'
        test_labeled_labeled_dir = 'data/test'

        self.results_dir = 'results'
        self.checkpoint_dir = 'weights'
        self.model_name = model_name

        # data augmentation
        self.mirror_aug = MirrorTransform()
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi/9, np.pi/9),
                                            angle_y=(-np.pi/9, np.pi/9),
                                            angle_z=(-np.pi/9, np.pi/9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # init the network
        self.Reger = Head(n_channels=n_channels)

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()

        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)

        self.stn = SpatialTransformer()
        self.rstn = Re_SpatialTransformer()
        self.softmax = nn.Softmax(dim=1)

        # init the data iterator
        train_labeled_unlabeled_dataset = DatasetFromFolder3D(train_labeled_unlabeled_dir, n_classes)
        self.dataloader_labeled_unlabeled = DataLoader(train_labeled_unlabeled_dataset, batch_size=batch_size, shuffle=True)
        train_unlabeled_unlabeled_dataset = DatasetFromFolder3D(train_unlabeled_unlabeled_dir, n_classes)
        self.dataloader_unlabeled_unlabeled = DataLoader(train_unlabeled_unlabeled_dataset, batch_size=batch_size, shuffle=True)
        test_labeled_labeled_dataset = DatasetFromFolder3D_Test(test_labeled_labeled_dir, n_classes)
        self.dataloader_labeled_labeled = DataLoader(test_labeled_labeled_dataset, batch_size=batch_size, shuffle=False)

        # define loss
        self.L_smooth = gradient_loss
        self.L_ncc = ncc_loss

        # define loss log
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_ncc_log = AverageMeter(name='L_ncc')

    def train_iterator(self, mi, fi, ml=None, fl=None):
        w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)

        # train Reger
        loss_s = self.L_smooth(flow)
        self.L_smooth_log.update(loss_s.data, mi.size(0))

        loss_ncc = torch.mean(self.L_ncc(w_m_to_f, fi))
        self.L_ncc_log.update(loss_ncc.data, mi.size(0))

        loss_Reg = loss_s + loss_ncc

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()

    def train_epoch(self, epoch, is_aug=True):
        self.Reger.train()
        for i in range(self.iters):
            rad = np.random.randint(low=0, high=2)
            if rad == 0:
                dataloader = self.dataloader_labeled_unlabeled
            else:
                dataloader = self.dataloader_unlabeled_unlabeled

            mov_img, fix_img, mov_lab, fix_lab = next(dataloader.__iter__())
            if len(mov_lab.data.numpy().shape) == 1:
                mov_lab = None
            if len(fix_lab.data.numpy().shape) == 1:
                fix_lab = None

            if torch.cuda.is_available():
                mov_img = mov_img.cuda()
                fix_img = fix_img.cuda()
                if mov_lab is not None:
                    mov_lab = mov_lab.cuda()
                if fix_lab is not None:
                    fix_lab = fix_lab.cuda()

            if is_aug:
                code_mir = self.mirror_aug.rand_code()
                code_spa = self.spatial_aug.rand_coords(mov_img.shape[2:])
                mov_img = self.mirror_aug.augment_mirroring(mov_img, code_mir)
                mov_img = self.spatial_aug.augment_spatial(mov_img, code_spa)
                fix_img = self.mirror_aug.augment_mirroring(fix_img, code_mir)
                fix_img = self.spatial_aug.augment_spatial(fix_img, code_spa)
                if mov_lab is not None:
                    mov_lab = self.mirror_aug.augment_mirroring(mov_lab, code_mir)
                    mov_lab = self.spatial_aug.augment_spatial(mov_lab, code_spa, mode='nearest')
                if fix_lab is not None:
                    fix_lab = self.mirror_aug.augment_mirroring(fix_lab, code_mir)
                    fix_lab = self.spatial_aug.augment_spatial(fix_lab, code_spa, mode='nearest')

            self.train_iterator(mov_img, fix_img, mov_lab, fix_lab)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_ncc_log.__str__()])
            print(res)

    def test_iterator(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Reg
            time_start = time.time()
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(mi, fi, ml, fl)
            time_end = time.time()
            print('time cost', time_end - time_start)
        return w_m_to_f, w_label_m_to_f, flow

    def test(self):
        self.Reger.eval()
        for i, (mi, fi, ml, fl, name) in enumerate(self.dataloader_labeled_labeled):
            name = name[0]
            print(name)
            if torch.cuda.is_available():
                mi = mi.cuda()
                fi = fi.cuda()
                ml = ml.cuda()
                fl = fl.cuda()

            w_m_to_f, w_label_m_to_f, flow = self.test_iterator(mi, fi, ml, fl)

            mi = mi.data.cpu().numpy()[0, 0]
            fi = fi.data.cpu().numpy()[0, 0]
            ml = np.argmax(ml.data.cpu().numpy()[0], axis=0)
            fl = np.argmax(fl.data.cpu().numpy()[0], axis=0)

            w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
            w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)
            flow = flow.data.cpu().numpy()[0]

            mi = mi.astype(np.float32)
            fi = fi.astype(np.float32)
            ml = ml.astype(np.float32)
            fl = fl.astype(np.float32)

            w_m_to_f = w_m_to_f.astype(np.float32)
            w_label_m_to_f = w_label_m_to_f.astype(np.float32)
            flow = flow.astype(np.float32)

            if not os.path.exists(join(self.results_dir, self.model_name, 'mi')):
                os.makedirs(join(self.results_dir, self.model_name, 'mi'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'fi')):
                os.makedirs(join(self.results_dir, self.model_name, 'fi'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'ml')):
                os.makedirs(join(self.results_dir, self.model_name, 'ml'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'fl')):
                os.makedirs(join(self.results_dir, self.model_name, 'fl'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'flow')):
                os.makedirs(join(self.results_dir, self.model_name, 'flow'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'w_m_to_f')):
                os.makedirs(join(self.results_dir, self.model_name, 'w_m_to_f'))
            if not os.path.exists(join(self.results_dir, self.model_name, 'w_label_m_to_f')):
                os.makedirs(join(self.results_dir, self.model_name, 'w_label_m_to_f'))

            mi.tofile(join(self.results_dir, self.model_name, 'mi', name[:-4]+'.raw'))
            fi.tofile(join(self.results_dir, self.model_name, 'fi', name[:-4]+'.raw'))
            ml.tofile(join(self.results_dir, self.model_name, 'ml', name[:-4]+'.raw'))
            fl.tofile(join(self.results_dir, self.model_name, 'fl', name[:-4]+'.raw'))

            w_m_to_f.tofile(join(self.results_dir, self.model_name, 'w_m_to_f', name[:-4]+'.raw'))
            w_label_m_to_f.tofile(join(self.results_dir, self.model_name, 'w_label_m_to_f', name[:-4]+'.raw'))
            flow.tofile(join(self.results_dir, self.model_name, 'flow', name[:-4]+'.raw'))
            print(name)

    def evaluate(self):
        DSC_R = np.zeros((self.n_classes, self.dataloader_labeled_labeled.__len__()))
        image_filenames = listdir(join(self.results_dir, self.model_name, 'w_m_to_f'))

        for i in range(len(image_filenames)):
            name = image_filenames[i]
            w_label_m_to_f = np.fromfile(join(self.results_dir, self.model_name, 'w_label_m_to_f', name), dtype=np.float32)
            w_label_m_to_f = to_categorical(w_label_m_to_f, self.n_classes)
            fl = np.fromfile(join(self.results_dir, self.model_name, 'fl', name), dtype=np.float32)
            fl = to_categorical(fl, self.n_classes)

            for c in range(self.n_classes):
                DSC_R[c, i] = dice(w_label_m_to_f[c], fl[c])

            print(name, DSC_R[1:, i])

        print(np.mean(DSC_R[1:, :], axis=1))
        print(np.mean(DSC_R[1:, :]))

    def checkpoint(self, epoch, k):
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)

    def load(self):
        self.Reger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, 'Reger_'+self.model_name, str(self.k))))

    def train(self):
        for epoch in range(self.epoches-self.k):
            self.L_smooth_log.reset()
            self.L_ncc_log.reset()

            self.train_epoch(epoch+self.k)
            if epoch % 20 == 0:
                self.checkpoint(epoch, self.k)
        self.checkpoint(self.epoches-self.k, self.k)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    RSTNet = RSeg()
    # RSTNet.load()
    RSTNet.train()
    RSTNet.test()
    RSTNet.evaluate()
