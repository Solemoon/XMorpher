import os
from os import listdir
from os.path import join

import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader

from models.UNet import UNet_reg, UNet_seg
from models.XMorpher import Head
from utils.STN import SpatialTransformer, Re_SpatialTransformer
from utils.augmentation import MirrorTransform, SpatialTransform
from utils.dataloader import DatasetFromFolder3D
from utils.dataloader_test import DatasetFromFolder3D_Test
from utils.losses import gradient_loss, ncc_loss, crossentropy, MSE, mask_crossentropy, dice_loss
from utils.utils import EMA, AverageMeter, to_categorical, dice
import numpy as np

class RSeg(object):
    def __init__(self, k=0, n_channels=1, n_classes=8, lr=1e-4, epoches=200, iters=200, batch_size=1, model_name='PC_XMorpher_heart_0'):
        super(RSeg, self).__init__()
        self.k = k
        self.n_classes = n_classes
        self.epoches = epoches
        self.iters=iters
        self.lr = lr
        train_labeled_unlabeled_dir = '/F/sjc/heart_CT_5/train_labeled_unlabeled'
        train_unlabeled_unlabeled_dir = '/F/sjc/heart_CT_5/train_unlabeled_unlabeled'
        test_labeled_labeled_dir = '/F/sjc/heart_CT_5/test'

        self.results_dir = 'results'
        self.checkpoint_dir = 'weights'
        self.model_name = model_name

        # data augmentation
        self.mirror_aug = MirrorTransform()
        self.spatial_aug = SpatialTransform(do_rotation=True,
                                            angle_x=(-np.pi / 9, np.pi / 9),
                                            angle_y=(-np.pi / 9, np.pi / 9),
                                            angle_z=(-np.pi / 9, np.pi / 9),
                                            do_scale=True,
                                            scale=(0.75, 1.25))

        # init the network
        self.Reger = Head(n_channels=n_classes-1)
        self.Seger = UNet_seg(n_channels=n_channels, n_classes=n_classes)

        if torch.cuda.is_available():
            self.Reger = self.Reger.cuda()
            self.Seger = self.Seger.cuda()

        self.optR = torch.optim.Adam(self.Reger.parameters(), lr=lr)
        self.optS = torch.optim.Adam(self.Seger.parameters(), lr=lr)

        self.stn = SpatialTransformer()
        self.rstn = Re_SpatialTransformer()
        self.softmax = nn.Softmax(dim=1)

        # init the data iterator
        train_labeled_unlabeled_dataset = DatasetFromFolder3D(train_labeled_unlabeled_dir, n_classes)
        self.dataloader_labeled_unlabeled = DataLoader(train_labeled_unlabeled_dataset, batch_size=batch_size, shuffle=True)
        train_unlabeled_unlabeled_dataset = DatasetFromFolder3D(train_unlabeled_unlabeled_dir, n_classes)
        self.dataloader_unlabeled_unlabeled = DataLoader(train_unlabeled_unlabeled_dataset, batch_size=batch_size, shuffle=True)
        Test_labeled_labeled_dataset = DatasetFromFolder3D_Test(test_labeled_labeled_dir, n_classes)
        self.dataloader_labeled_labeled = DataLoader(Test_labeled_labeled_dataset, batch_size=batch_size, shuffle=False)

        # define loss
        self.L_smooth = gradient_loss
        self.L_ncc = ncc_loss
        self.L_seg = crossentropy

        # define loss log
        self.L_smooth_log = AverageMeter(name='L_smooth')
        self.L_ncc_log = AverageMeter(name='L_ncc')
        self.L_w_log = AverageMeter(name='L_w')
        self.L_anchor_log = AverageMeter(name='L_anchor')

    def train_iterator_stage1(self, mi, fi, ml=None, fl=None):
        # train Seger
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False below in netG update

        s_f = self.softmax(self.Seger(fi))
        s_m = self.softmax(self.Seger(mi))

        if ml is not None and fl is not None:
            loss_anchor1 = self.L_seg(s_f, fl)
            self.L_anchor_log.update(loss_anchor1.data, fi.shape[0])
            loss_anchor2 = self.L_seg(s_m, ml)
            self.L_anchor_log.update(loss_anchor2.data, fi.shape[0])
            loss_seg = loss_anchor1 + loss_anchor2
            loss_seg.backward()
            self.optS.step()
            self.Seger.zero_grad()
            self.optS.zero_grad()
        elif ml is not None and fl is None:
            loss_anchor2 = self.L_seg(s_m, ml)
            self.L_anchor_log.update(loss_anchor2.data, fi.shape[0])
            loss_seg = loss_anchor2
            loss_seg.backward()
            self.optS.step()
            self.Seger.zero_grad()
            self.optS.zero_grad()
        elif ml is None and fl is not None:
            loss_anchor1 = self.L_seg(s_f, fl)
            self.L_anchor_log.update(loss_anchor1.data, fi.shape[0])

            loss_seg = loss_anchor1
            loss_seg.backward()
            self.optS.step()
            self.Seger.zero_grad()
            self.optS.zero_grad()

        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = True  # they are set to False below in netG update
        s_f = s_f.detach()
        s_m = s_m.detach()
        s_f = torch.argmax(s_f, dim=1, keepdim=True)
        s_f = [torch.where(s_f==i, torch.full_like(s_f, 1), torch.full_like(s_f, 0)) for i in range(self.n_classes)]
        s_f = torch.cat(s_f, dim=1)
        s_m = torch.argmax(s_m, dim=1, keepdim=True)
        s_m = [torch.where(s_m == i, torch.full_like(s_m, 1), torch.full_like(s_m, 0)) for i in range(self.n_classes)]
        s_m = torch.cat(s_m, dim=1)
        r_fi = s_f * fi
        r_fi = r_fi[:, 1:, :, :, :]
        r_mi = s_m * mi
        r_mi = r_mi[:, 1:, :, :, :]

        # train Reger
        w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(r_mi, r_fi, ml, fl)

        loss_s = self.L_smooth(flow)
        self.L_smooth_log.update(loss_s.data, mi.size(0))
        loss_ncc = 0
        for i in range(self.n_classes-1):
            loss_ncc += self.L_ncc(w_m_to_f[:, i:i+1, :, :, :], r_fi[:, i:i+1, :, :, :])

        self.L_ncc_log.update(loss_ncc.data, mi.size(0))

        loss_Reg = loss_s + loss_ncc

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()

    def train_iterator_stage2(self, mi, fi, ml=None, fl=None):
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = True  # they are set to False below in netG update
        with torch.no_grad():
            s_f = self.softmax(self.Seger(fi))
            s_m = self.softmax(self.Seger(mi))
        s_f = torch.argmax(s_f, dim=1, keepdim=True)
        s_f = [torch.where(s_f == i, torch.full_like(s_f, 1), torch.full_like(s_f, 0)) for i in range(self.n_classes)]
        s_f = torch.cat(s_f, dim=1)
        s_m = torch.argmax(s_m, dim=1, keepdim=True)
        s_m = [torch.where(s_m == i, torch.full_like(s_m, 1), torch.full_like(s_m, 0)) for i in range(self.n_classes)]
        s_m = torch.cat(s_m, dim=1)
        r_fi = s_f * fi
        r_fi = r_fi[:, 1:, :, :, :]
        r_mi = s_m * mi
        r_mi = r_mi[:, 1:, :, :, :]

        # train Reger
        w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(r_mi, r_fi, ml, fl)
        loss_s = self.L_smooth(flow)
        self.L_smooth_log.update(loss_s.data, mi.size(0))

        loss_ncc = 0
        for i in range(self.n_classes - 1):
            loss_ncc += self.L_ncc(w_m_to_f[:, i:i + 1, :, :, :], r_fi[:, i:i+1, :, :, :])

        self.L_ncc_log.update(loss_ncc.data, mi.size(0))

        loss_Reg = loss_s + loss_ncc

        loss_Reg.backward()
        self.optR.step()
        self.Reger.zero_grad()
        self.optR.zero_grad()

        # RT
        for p in self.Seger.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in self.Reger.parameters():  # reset requires_grad             -
            p.requires_grad = False  # they are set to False below in netG update
        flow = flow.detach()

        if ml is not None and fl is None:
            w_m_to_f = self.stn(mi, flow)
            w_label_m_to_f = self.stn(ml, flow)
            s_w_m_to_f = self.softmax(self.Seger(w_m_to_f))
            s_f = self.softmax(self.Seger(fi))

            loss_w = self.L_seg(s_f, w_label_m_to_f)
            self.L_w_log.update(loss_w.data, fi.shape[0])

            loss_anchor2 = self.L_seg(s_w_m_to_f, w_label_m_to_f)
            self.L_anchor_log.update(loss_anchor2.data, fi.shape[0])
            loss_seg = loss_anchor2 + 0.5*loss_w
            loss_seg.backward()
            self.optS.step()
            self.Seger.zero_grad()
            self.optS.zero_grad()
        elif ml is None and fl is not None:
            w_m_to_f = self.stn(mi, flow)
            s_w_m_to_f = self.softmax(self.Seger(w_m_to_f))
            s_f = self.softmax(self.Seger(fi))

            loss_anchor1 = self.L_seg(s_f, fl)
            self.L_anchor_log.update(loss_anchor1.data, fi.shape[0])

            loss_w = self.L_seg(s_w_m_to_f, fl)

            loss_seg = loss_anchor1 + 0.5*loss_w
            loss_seg.backward()
            self.optS.step()
            self.Seger.zero_grad()
            self.optS.zero_grad()

    def train_epoch_stage1(self, epoch, is_aug=True):
        self.Seger.train()
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

            self.train_iterator_stage1(mov_img, fix_img, mov_lab, fix_lab)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_ncc_log.__str__(),
                             self.L_anchor_log.__str__()])
            print("Stage 1:", res)

    def train_epoch_stage2(self, epoch, is_aug=True):
        self.Seger.train()
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

            self.train_iterator_stage2(mov_img, fix_img, mov_lab, fix_lab)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_smooth_log.__str__(),
                             self.L_ncc_log.__str__(),
                             self.L_w_log.__str__(),
                             self.L_anchor_log.__str__()])
            print("Stage 2:", res)


    def test_iterator(self, mi, fi, ml=None, fl=None):
        with torch.no_grad():
            # Seg
            s_m = self.softmax(self.Seger(mi))
            s_f = self.softmax(self.Seger(fi))

            # Reg
            r_fi = s_f * fi
            r_fi = r_fi[:, 1:, :, :, :]
            r_mi = s_m * mi
            r_mi = r_mi[:, 1:, :, :, :]
            w_m_to_f, w_f_to_m, w_label_m_to_f, w_label_f_to_m, flow = self.Reger(r_mi, r_fi, ml, fl)

            w_m_to_f = self.stn(mi, flow)
            w_label_m_to_f = self.stn(ml, flow)

        return w_m_to_f, w_label_m_to_f, s_m, s_f, flow

    def test(self):
        self.Seger.eval()
        self.Reger.eval()
        for i, (mi, fi, ml, fl, name) in enumerate(self.dataloader_labeled_labeled):
            name = name[0]
            if torch.cuda.is_available():
                mi = mi.cuda()
                fi = fi.cuda()
                ml = ml.cuda()
                fl = fl.cuda()

            w_m_to_f, w_label_m_to_f, s_m, s_f, flow = self.test_iterator(mi, fi, ml, fl)

            mi = mi.data.cpu().numpy()[0, 0]
            fi = fi.data.cpu().numpy()[0, 0]
            ml = np.argmax(ml.data.cpu().numpy()[0], axis=0)
            fl = np.argmax(fl.data.cpu().numpy()[0], axis=0)
            flow = flow.data.cpu().numpy()[0]

            w_m_to_f = w_m_to_f.data.cpu().numpy()[0, 0]
            w_label_m_to_f = np.argmax(w_label_m_to_f.data.cpu().numpy()[0], axis=0)
            s_m = np.argmax(s_m.data.cpu().numpy()[0], axis=0)
            s_f = np.argmax(s_f.data.cpu().numpy()[0], axis=0)

            mi = mi.astype(np.float32)
            fi = fi.astype(np.float32)
            ml = ml.astype(np.float32)
            fl = fl.astype(np.float32)

            w_m_to_f = w_m_to_f.astype(np.float32)
            w_label_m_to_f = w_label_m_to_f.astype(np.float32)
            s_m = s_m.astype(np.float32)
            s_f = s_f.astype(np.float32)
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
            if not os.path.exists(join(self.results_dir, self.model_name, 's_m')):
                os.makedirs(join(self.results_dir, self.model_name, 's_m'))
            if not os.path.exists(join(self.results_dir, self.model_name, 's_f')):
                os.makedirs(join(self.results_dir, self.model_name, 's_f'))

            mi.tofile(join(self.results_dir, self.model_name, 'mi', name[:-4]+'.raw'))
            fi.tofile(join(self.results_dir, self.model_name, 'fi', name[:-4]+'.raw'))
            ml.tofile(join(self.results_dir, self.model_name, 'ml', name[:-4]+'.raw'))
            fl.tofile(join(self.results_dir, self.model_name, 'fl', name[:-4]+'.raw'))

            w_m_to_f.tofile(join(self.results_dir, self.model_name, 'w_m_to_f', name[:-4]+'.raw'))
            w_label_m_to_f.tofile(join(self.results_dir, self.model_name, 'w_label_m_to_f', name[:-4]+'.raw'))
            s_m.tofile(join(self.results_dir, self.model_name, 's_m', name[:-4]+'.raw'))
            s_f.tofile(join(self.results_dir, self.model_name, 's_f', name[:-4]+'.raw'))
            flow.tofile(join(self.results_dir, self.model_name, 'flow', name[:-4]+'.raw'))
            print(name)

    def evaluate(self):
        DSC_R = np.zeros((self.n_classes, self.dataloader_labeled_labeled.__len__()))
        DSC_S = np.zeros((self.n_classes, self.dataloader_labeled_labeled.__len__()))
        image_filenames = listdir(join(self.results_dir, self.model_name, 's_f'))

        for i in range(len(image_filenames)):
            name = image_filenames[i]
            w_label_m_to_f = np.fromfile(join(self.results_dir, self.model_name, 'w_label_m_to_f', name), dtype=np.float32)
            w_label_m_to_f = to_categorical(w_label_m_to_f, self.n_classes)
            fl = np.fromfile(join(self.results_dir, self.model_name, 'fl', name), dtype=np.float32)
            fl = to_categorical(fl, self.n_classes)
            ml = np.fromfile(join(self.results_dir, self.model_name, 'ml', name), dtype=np.float32)
            ml = to_categorical(ml, self.n_classes)
            s_m = np.fromfile(join(self.results_dir, self.model_name, 's_m', name), dtype=np.float32)
            s_m = to_categorical(s_m, self.n_classes)

            for c in range(self.n_classes):
                DSC_R[c, i] = dice(w_label_m_to_f[c], fl[c])
                DSC_S[c, i] = dice(s_m[c], ml[c])

            print(name, DSC_S[1:, i], DSC_R[1:, i])

        print(np.mean(DSC_S[1:, :], axis=1), np.mean(DSC_R[1:, :], axis=1))
        # print(np.mean(DSC_S[1:, :]), np.mean(DSC_R[1:, :]))
        print(np.mean(DSC_S[1:, :]), np.std(np.mean(DSC_S[1:, :], axis=0)))
        print(np.mean(DSC_R[1:, :]), np.std(np.mean(DSC_R[1:, :], axis=0)))

    def checkpoint(self, epoch, k, stage='Stage1'):
        torch.save(self.Seger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, stage+'_Seger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)
        torch.save(self.Reger.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, stage+'_Reger_'+self.model_name, epoch+k),
                   _use_new_zipfile_serialization=False)

    def load(self, stage='Stage2'):
        self.Reger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, stage+'_Reger_'+self.model_name, str(self.k))))
        self.Seger.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, stage+'_Seger_' + self.model_name, str(self.k))))

    def train(self, stage1=True, stage2=True):
        if stage1:
            for epoch in range(self.epoches-self.k):
                self.L_smooth_log.reset()
                self.L_ncc_log.reset()
                self.L_w_log.reset()
                self.L_anchor_log.reset()
                self.train_epoch_stage1(epoch+self.k)
                if epoch % 20 == 0:
                    self.checkpoint(epoch, self.k, stage='Stage1')

            self.checkpoint(self.epoches-self.k, self.k, stage='Stage1')

        if stage2:
            for epoch in range(self.epoches - self.k):
                self.L_smooth_log.reset()
                self.L_ncc_log.reset()
                self.L_w_log.reset()
                self.L_anchor_log.reset()
                self.train_epoch_stage2(epoch + self.k)
                if epoch % 20 == 0:
                    self.checkpoint(epoch, self.k, stage='Stage2')

            self.checkpoint(self.epoches-self.k, self.k, stage='Stage2')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    RSTNet = RSeg()
    # RSTNet.load(stage='Stage2')
    RSTNet.train(stage1=True, stage2=True)
    RSTNet.test()
    RSTNet.evaluate()


# export CUDA_VISIBLE_DEVICES=2
# nohup python semi_pcreg.py > run_pcreg_cross455.log 2>&1 &