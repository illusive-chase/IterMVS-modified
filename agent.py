import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from .datasets import MVSDataset, read_pair_file
from .models import Pipeline
from .utils import compare_pairs, tensor2numpy, tocuda
import cv2
import gc
from collections import namedtuple

CUDAView = namedtuple('CUDAView', ['intrinsics', 'extrinsics', 'intrinsics_inv', 'extrinsics_inv', 'depth_est'])

class IncrementalIterMVSAgent:
    def __init__(self, folder, cuda, config={}):
        self.cuda = cuda
        self.folder = folder
        self.device = torch.device(f'cuda:{cuda}')
        self.config = config

        self.redirect = config.get('redirect', '/dev/stdout')
        self.batch_size = config.get('batch_size', 8)
        self.n_views = config.get('n_views', 5)
        self.img_wh = config.get('img_wh', (640, 480))
        self.loadckpt = config.get('loadckpt', './checkpoints/blendedmvs/model_000015.ckpt')
        self.iteration = config.get('iteration', 4)
        self.store_color = config.get('store_color', True)
        self.store_depth = config.get('store_depth', False)
        self.store_feature = config.get('store_feature', False)
        self.store_confidence = config.get('store_confidence', False)
        self.photo_thres = config.get('photo_thres', 0.3)
        self.geo_pixel_thres = config.get('geo_pixel_thres', 1)
        self.geo_depth_thres = config.get('geo_depth_thres', 0.01)
        self.geo_mask_thres = config.get('geo_mask_thres', 3)
        self.cropping_aabb = config.get('cropping_aabb', np.array([-np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf]))

        assert not self.store_feature or self.store_confidence

        self._cache_mesh_grid_WxH = None
        self._cache_pad_3x3_to_4x4 = None
        self._cache_pad_2xN_to_3xN = None
        self._cache_pad_3xN_to_4xN = None
        
        with self.open_stream() as stream:
            stream.write("loading model {}\n".format(self.loadckpt))
        self.model = Pipeline(iteration=self.iteration, test=True).to(self.device)
        self.model.load_state_dict(torch.load(self.loadckpt))
        self.model.eval()
        self.dataset = MVSDataset(folder=self.folder, n_views=self.n_views, img_wh=self.img_wh)

    def extract_feature(self):
        imgs = np.stack([view_data.LOD['level_0'] for view_data in self.dataset.view_data.values()]).transpose([0, 3, 1, 2])
        inds = [vid for vid in self.dataset.view_data.keys()]
        with self.open_stream() as stream:
            stream.write('Extracting...\n')
            extract_batch_size = self.batch_size * 2
            for i in range(0, imgs.shape[0], extract_batch_size):
                start_time = time.time()
                sample_cuda = torch.from_numpy(imgs[i:i+extract_batch_size]).to(self.device).unsqueeze(1).contiguous()
                outputs = tensor2numpy(self.model.extract_feature(sample_cuda))
                # outputs = nn.functional.interpolate(self.model.extract_feature(sample_cuda)['level1'][0], scale_factor=0.25, mode='bilinear', align_corners=False, recompute_scale_factor=True).cpu().numpy()
                stream.write('Iter {}/{}, time = {:.3f}\n'.format(i // extract_batch_size, (imgs.shape[0] + extract_batch_size - 1) // extract_batch_size, time.time() - start_time))
                
                for j in range(0, min(extract_batch_size, imgs.shape[0] - i)):
                    yield inds[i + j], { k : v[0][j] for k, v in outputs.items() }

    def save_depth(self, stream, feature_pool=None):
        with torch.no_grad():
            self.dataset.precalculate_feature = feature_pool is not None

            TestImgLoader = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=0, drop_last=False)
        
            stream.write('Inferring...\n')
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                if feature_pool is not None:
                    view_ids = sample["view_ids"]
                    sample["features"] = {
                        k: torch.from_numpy(v.reshape(*view_ids.size(), *v.shape[1:]))
                        for k, v in feature_pool.get_features(view_ids.reshape(-1)).items()
                    }
                else:
                    sample["features"] = {}
                sample_cuda = tocuda(sample, self.device)
                outputs = tensor2numpy(self.model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_min"], sample_cuda["depth_max"], sample_cuda["features"]))
                del sample_cuda
                stream.write('Iter {}/{}, time = {:.3f}\n'.format(batch_idx, len(TestImgLoader), time.time() - start_time))

                for view_id, depth_est, confidence in zip(sample["view_id"], outputs["depths_upsampled"], outputs["confidence_upsampled"]):
                    view_id = view_id.item()
                    depth_est = np.squeeze(depth_est, 0)
                    self.dataset.view_data[view_id].depth[0] = depth_est
                    self.dataset.view_data[view_id].confidence[0] = np.squeeze(confidence, 0)

    def get_cuda_view(self, idx):
        cuda_view = self.cache.get(idx)
        if cuda_view is None:
            view_data = self.dataset.view_data[idx]
            cuda_view = CUDAView(
                intrinsics=torch.from_numpy(view_data.intrinsics).to(self.device),
                extrinsics=torch.from_numpy(view_data.extrinsics).to(self.device),
                intrinsics_inv=torch.from_numpy(view_data.inv_intrinsics).to(self.device),
                extrinsics_inv=torch.from_numpy(view_data.inv_extrinsics).to(self.device),
                depth_est=view_data.depth[0]
            )
            self.cache[idx] = cuda_view
        return cuda_view

    def reproject_with_depth(self, ref_idx, src_idx, depth_ref, depth_src, ref_view_xyz1):
        ref_view = self.get_cuda_view(ref_idx)
        src_view = self.get_cuda_view(src_idx)
        K_xyz_src = ((self._pad_3x3_to_4x4(src_view.intrinsics) @ src_view.extrinsics @ ref_view.extrinsics_inv) @ ref_view_xyz1)[:3]
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]
        xy_src_cpu = xy_src.cpu().numpy()
        x_src = xy_src_cpu[0].reshape(self.img_wh[1], self.img_wh[0])
        y_src = xy_src_cpu[1].reshape(self.img_wh[1], self.img_wh[0])
        assert x_src.dtype == np.float32
        assert y_src.dtype == np.float32
        sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR).reshape(1, -1)
        xyz1_src = self._pad_3xN_to_4xN(src_view.intrinsics_inv @ (self._pad_2xN_to_3xN(xy_src) * torch.from_numpy(sampled_depth_src).to(self.device)))
        xyz_reprojected = ((ref_view.extrinsics @ src_view.extrinsics_inv) @ xyz1_src)[:3]
        depth_reprojected = xyz_reprojected[2].view(depth_src.shape)
        K_xyz_reprojected = ref_view.intrinsics @ xyz_reprojected
        xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-6)
        x_reprojected = xy_reprojected[0].view(depth_src.shape)
        y_reprojected = xy_reprojected[1].view(depth_src.shape)
        return depth_reprojected, x_reprojected, y_reprojected

    def check_geometric_consistency(self, ref_idx, src_idx, depth_ref, depth_src, ref_view_xyz1):
        x_ref, y_ref = self._mesh_grid_WxH(as_xy1=False)
        depth_reprojected, x2d_reprojected, y2d_reprojected = self.reproject_with_depth(ref_idx, src_idx, depth_ref, depth_src, ref_view_xyz1)

        mask = (torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2) < self.geo_pixel_thres) & \
                ((torch.abs(depth_reprojected - depth_ref) / depth_ref) < self.geo_depth_thres)
        depth_reprojected[~mask] = 0

        return mask, depth_reprojected

    def filter_depth(self, stream, fusion_pairs):

        aabb = torch.from_numpy(self.cropping_aabb).to(self.device)
        xyz_min = aabb[[0,2,4]]
        xyz_max = aabb[[1,3,5]]

        for ref_view, src_views in fusion_pairs:
            confidence = torch.from_numpy(self.dataset.view_data[ref_view].confidence[0]).to(self.device)
            photo_mask = confidence > self.photo_thres

            all_srcview_depth_ests = 0
            geo_mask_sum = 0

            ref_cuda_view = self.get_cuda_view(ref_view)
            ref_depth_est = torch.from_numpy(ref_cuda_view.depth_est).to(self.device)
            ref_view_xyz1 = self._pad_3xN_to_4xN(ref_cuda_view.intrinsics_inv @ (self._mesh_grid_WxH(as_xy1=True) * ref_depth_est.view(1, -1))).clone()

            for src_view in src_views:
                src_depth_est = self.dataset.view_data[src_view].depth[0]
                
                geo_mask, depth_reprojected = self.check_geometric_consistency(
                    ref_idx=ref_view,
                    src_idx=src_view,
                    depth_ref=ref_depth_est,
                    depth_src=src_depth_est,
                    ref_view_xyz1=ref_view_xyz1
                )
                geo_mask_sum = geo_mask_sum + geo_mask.int()
                all_srcview_depth_ests = all_srcview_depth_ests + depth_reprojected

            depth_est_averaged = (all_srcview_depth_ests + ref_depth_est) / (geo_mask_sum + 1)
            geo_mask = geo_mask_sum >= self.geo_mask_thres

            valid_points = photo_mask & geo_mask
            stream.write("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}\n".format(
                self.folder,
                ref_view,
                geo_mask.float().mean().item(),
                photo_mask.float().mean().item(),
                valid_points.float().mean().item()
            ))
            x, y = self._mesh_grid_WxH(as_xy1=False)
            x, y, depth, confidence = x[valid_points], y[valid_points], depth_est_averaged[valid_points], confidence[valid_points]
            
            view = self.get_cuda_view(ref_view)
            xyz_ref = view.intrinsics_inv @ (torch.cat((x.view(1, -1), y.view(1, -1), torch.ones_like(x).view(1, -1)), dim=0) * depth)
            xyz_world = (view.extrinsics_inv @ torch.cat((xyz_ref, torch.ones_like(x).float().view(1, -1)), dim=0))[:3].transpose(1, 0)
            cropping_mask = (xyz_world > xyz_min).all(1) & (xyz_world < xyz_max).all(1)

            point_data = self.dataset.view_data[ref_view].points
            point_data.xyz[0] = xyz_world[cropping_mask].cpu().numpy()
            if self.store_color:
                point_data.rgb[0] = self.dataset.view_data[ref_view].LOD['level_0'][valid_points.cpu().numpy()][cropping_mask.cpu().numpy()]
            if self.store_confidence:
                point_data.conf[0] = confidence[cropping_mask].cpu().numpy()
            # if self.store_feature:
            #     feature_indices = (valid_points.nonzero(as_tuple=False)[cropping_mask, :] // 8).cpu().numpy()
            #     point_data.feature[0] = self.dataset.view_data[ref_view].features[0][feature_indices[:, 0], feature_indices[:, 1]]
            

    def _mesh_grid_WxH(self, as_xy1):
        if self._cache_mesh_grid_WxH is None:
            xs, ys = np.meshgrid(np.arange(0, self.img_wh[0], dtype=np.int32), np.arange(0, self.img_wh[1], dtype=np.int32))
            self._cache_mesh_grid_WxH = torch.ones((3, self.img_wh[1], self.img_wh[0]), device=self.device)
            self._cache_mesh_grid_WxH[0].copy_(torch.from_numpy(xs))
            self._cache_mesh_grid_WxH[1].copy_(torch.from_numpy(ys))
        if as_xy1:
            return self._cache_mesh_grid_WxH.view(3, -1)
        return self._cache_mesh_grid_WxH[0], self._cache_mesh_grid_WxH[1]

    def _pad_3x3_to_4x4(self, intrinsics):
        if self._cache_pad_3x3_to_4x4 is None:
            self._cache_pad_3x3_to_4x4 = torch.eye(4, device=self.device)
        self._cache_pad_3x3_to_4x4[:3, :3] = intrinsics
        return self._cache_pad_3x3_to_4x4

    def _pad_3xN_to_4xN(self, coords):
        if self._cache_pad_3xN_to_4xN is None:
            self._cache_pad_3xN_to_4xN = torch.ones((4, self.img_wh[0] * self.img_wh[1]), device=self.device)
        self._cache_pad_3xN_to_4xN[0:3, :] = coords
        return self._cache_pad_3xN_to_4xN

    def _pad_2xN_to_3xN(self, coords):
        if self._cache_pad_2xN_to_3xN is None:
            self._cache_pad_2xN_to_3xN = torch.ones((3, self.img_wh[0] * self.img_wh[1]), device=self.device)
        self._cache_pad_2xN_to_3xN[0:2, :] = coords
        return self._cache_pad_2xN_to_3xN

    def open_stream(self):
        return open(self.redirect, ('w' if self.redirect.startswith('/dev/') else 'a'))

    def reset(self):
        self.cache = {}
        self.pair_data = []
        gc.collect()

    def step(self, pair_data, feature_pool=None):
        estimation_pairs, fusion_pairs = compare_pairs(self.pair_data, pair_data)
        self.pair_data = pair_data
        self.dataset.update(estimation_pairs)
        with self.open_stream() as stream:
            self.save_depth(stream, feature_pool=feature_pool)
            self.filter_depth(stream, fusion_pairs)

    def extract_point_cloud(self):
        xyzs = []
        rgbs = []
        for view in self.dataset.view_data.values():
            if view.points.xyz[0] is None:
                continue
            assert ((not self.store_color) or (view.points.rgb[0] is not None))
            xyzs.append(view.points.xyz[0])
            if self.store_color:
                rgbs.append(view.points.rgb[0])
        with self.open_stream() as stream:
            stream.write('Total {} points.\n'.format(sum([v.shape[0] for v in xyzs])))
        if self.store_color:
            return {
                'xyz': np.concatenate(xyzs, axis=0),
                'rgb': np.concatenate(rgbs, axis=0)
            }
        else:
            return { 'xyz': np.concatenate(xyzs, axis=0) }

