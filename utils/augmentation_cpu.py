import numpy as np
from batchgenerators.augmentations.utils import interpolate_img, create_zero_centered_coordinate_mesh, \
    elastic_deform_coordinates, rotate_coords_3d, scale_coords


class MirrorTransform(object):
    def augment_mirroring(self, data, code=(1, 1, 1)):
        if code[0] == 1:
            data[:] = data[::-1]
        if code[1] == 1:
            data[:, :] = data[:, ::-1]
        if code[2] == 1:
            data[:, :, :] = data[:, :, ::-1]
        return data

    def rand_code(self):
        code = []
        for i in range(3):
            if np.random.uniform() < 0.5:
                code.append(1)
            else:
                code.append(0)
        return code


class SpatialTransform(object):
    def __init__(self, do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='nearest', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=0):
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg

    def augment_spatial(self, data, coords, is_label=False):
        if is_label:
            order = self.order_seg
            border_mode = self.border_mode_seg
            border_cval = self.border_cval_seg
        else:
            order= self.order_data
            border_mode = self.border_mode_data
            border_cval = self.border_cval_data
        data = interpolate_img(data, coords, order, border_mode, cval=border_cval)
        return data

    def rand_coords(self, patch_size):
        coords = create_zero_centered_coordinate_mesh(patch_size)
        if self.do_rotation:
            a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
            coords = rotate_coords_3d(coords, a_x, a_y, a_z)

        if self.do_scale:
            sc = np.random.uniform(self.scale[0], self.scale[1])
            coords = scale_coords(coords, sc)

        ctr = np.asarray([patch_size[0]//2, patch_size[1]//2, patch_size[2]//2])
        coords += ctr[:, np.newaxis, np.newaxis, np.newaxis]
        return coords
