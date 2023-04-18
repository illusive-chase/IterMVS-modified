from torch.utils.data import Dataset
from .data_io import read_cam_file, read_img
import os
import numpy as np
from collections import namedtuple

ViewData = namedtuple('ViewData', [
    'extrinsics',
    'intrinsics',
    'inv_extrinsics',
    'inv_intrinsics',
    'depth_max',
    'depth_min',
    'idx',
    'depth',
    'confidence',
    'LOD',
    'points'
])
PointsData = namedtuple('PointsData', ['rgb', 'xyz', 'conf'])


class MVSDataset(Dataset):
    def __init__(self, folder, n_views=5, img_wh=(640,480)):
        self.levels = 4
        self.folder = folder
        self.img_wh = img_wh
        self.metas = []
        self.n_views = n_views
        self.view_data = {}

    def load(self, view_id):
        if view_id not in self.view_data:
            intrinsics, extrinsics, depth_min, depth_max = read_cam_file(os.path.join(self.folder, 'cams_1', '{:08d}_cam.txt'.format(view_id)))
            LOD, original_h, original_w = read_img(os.path.join(self.folder, 'images', '{:08d}.jpg'.format(view_id)), self.img_wh[1], self.img_wh[0])
            intrinsics[0] *= self.img_wh[0]/original_w
            intrinsics[1] *= self.img_wh[1]/original_h
            self.view_data[view_id] = ViewData(
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                inv_extrinsics=np.linalg.inv(extrinsics),
                inv_intrinsics=np.linalg.inv(intrinsics),
                depth_max=depth_max,
                depth_min=depth_min,
                idx=view_id,
                depth=[None],
                confidence=[None],
                LOD=LOD,
                points=PointsData(
                    rgb=[None],
                    xyz=[None],
                    conf=[None]
                )
            )

    def update(self, estimation_pairs):
        self.metas = estimation_pairs
        view_ids = set()
        for ref_view, src_views in self.metas:
            view_ids = view_ids | set([ref_view] + src_views[:self.n_views-1])
        for view_id in view_ids:
            if view_id not in self.view_data:
                self.load(view_id)
            

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        ref_view, src_views = self.metas[idx]
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views-1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []

        # depth = None
        depth_min = None
        depth_max = None

        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):

            view_data = self.view_data[vid]
            imgs_0.append(view_data.LOD['level_0'])
            imgs_1.append(view_data.LOD['level_1'])
            imgs_2.append(view_data.LOD['level_2'])
            imgs_3.append(view_data.LOD['level_3'])

            intrinsics = view_data.intrinsics.copy()
            extrinsics = view_data.extrinsics
            depth_min_ = view_data.depth_min
            depth_max_ = view_data.depth_max

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat if i > 0 else np.linalg.inv(proj_mat))

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat if i > 0 else np.linalg.inv(proj_mat))

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat if i > 0 else np.linalg.inv(proj_mat))

            proj_mat = extrinsics.copy()
            intrinsics[:2,:] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat if i > 0 else np.linalg.inv(proj_mat))

            if i == 0:  # reference view
                depth_min = depth_min_
                depth_max = depth_max_

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        imgs = {}
        imgs['level_0'] = imgs_0
        imgs['level_1'] = imgs_1
        imgs['level_2'] = imgs_2
        imgs['level_3'] = imgs_3
        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        proj={}
        proj['level_3']=proj_matrices_3
        proj['level_2']=proj_matrices_2
        proj['level_1']=proj_matrices_1
        proj['level_0']=proj_matrices_0

        return {"imgs": imgs,                   # N*3*H0*W0
                "proj_matrices": proj, # N*4*4
                "depth_min": depth_min,         # scalar
                "depth_max": depth_max,
                "view_id": ref_view
                }  
