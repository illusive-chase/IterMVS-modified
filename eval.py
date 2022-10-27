import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
from .datasets import MVSDataset, read_pfm, save_pfm
from .models import Pipeline
from .utils import print_args, tensor2numpy, tocuda, compare_pairs, write_ply
import sys
import cv2
from PIL import Image

cudnn.benchmark = True

def main(workdir,
         batch_size,
         n_views,
         img_wh,
         loadckpt,
         iteration,
         device,
         cuda,
         color,
         output,
         photo_thres,
         geo_pixel_thres,
         geo_depth_thres,
         geo_mask_thres,
         recompute
    ):

    def lazy(func):
        same_dict = {}
        def wrapper(*arglst):
            key = ';'.join([str(arg) for arg in arglst])
            if key not in same_dict:
                same_dict[key] = func(*arglst)
            return same_dict[key]
        return wrapper

    # read intrinsics and extrinsics
    def read_camera_parameters(filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        
        return intrinsics, extrinsics


    # read an image
    def read_img(filename, img_wh):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        original_h, original_w, _ = np_img.shape
        np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
        return np_img, original_h, original_w

    def read_size(filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        return np_img.shape[:2]


    # save a binary mask
    def save_mask(filename, mask):
        assert mask.dtype == np.bool_
        mask = mask.astype(np.uint8) * 255
        Image.fromarray(mask).save(filename)

    def save_depth_img(filename, depth):
        # assert mask.dtype == np.bool
        depth = depth.astype(np.float32) * 255
        Image.fromarray(depth).save(filename)


    def read_pair_file(filename):
        data = []
        with open(filename) as f:
            num_viewpoint = int(f.readline())
            # 49 viewpoints
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                if len(src_views) != 0:
                    data.append((ref_view, src_views))
        return data


    # run MVS model to save depth maps
    def save_depth(estimation_pairs):
        # dataset, dataloader
        test_dataset = MVSDataset(workdir, estimation_pairs, n_views, img_wh)
        TestImgLoader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, drop_last=False)

        # model
        model = Pipeline(iteration=iteration, test=True)
        model = nn.DataParallel(model)
        model.cuda()

        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        model.eval()
        
        with torch.no_grad():
            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                sample_cuda = tocuda(sample)
                outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                            sample_cuda["depth_min"], sample_cuda["depth_max"])

                outputs = tensor2numpy(outputs)
                del sample_cuda
                print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
                filenames = sample["filename"]

                # save depth maps and confidence maps
                for filename, depth_est, confidence in zip(filenames, outputs["depths_upsampled"], outputs["confidence_upsampled"]):
                    depth_filename = os.path.join(workdir, filename.format('depth_est', '.pfm'))
                    confidence_filename = os.path.join(workdir, filename.format('confidence', '.pfm'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    # save depth maps
                    depth_est = np.squeeze(depth_est, 0)
                    save_pfm(depth_filename, depth_est)
                    # save confidence maps
                    confidence = np.squeeze(confidence, 0)
                    save_pfm(confidence_filename, confidence)

    @lazy
    def get_ones(shape):
        return np.ones(shape, dtype=np.float32)

    @lazy
    def get_eye(n):
        return np.eye(n, dtype=np.float32)

    def fast_vstack_1(coords):
        ones = get_ones((coords.shape[0] + 1, *coords.shape[1:]))
        ones[:coords.shape[0]] = coords
        return ones

    def fast_padding(intrinsics):
        eyes = get_eye(4)
        eyes[:3, :3] = intrinsics
        return eyes

    @lazy
    def get_grid(W, H):
        return np.meshgrid(np.arange(0, W, dtype=np.int32), np.arange(0, H, dtype=np.int32))

    @lazy
    def get_xy1(W, H):
        xs, ys = get_grid(W, H)
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        return np.vstack((xs, ys, np.ones_like(xs)))

    @lazy
    def get_cuda_xy1(W, H):
        return torch.from_numpy(get_xy1(W, H)).to(device)

    @lazy
    def get_cuda_eye(n):
        return torch.from_numpy(np.eye(n, dtype=np.float32)).to(device)

    @lazy
    def get_cuda_ones(shape):
        return torch.from_numpy(np.ones(shape, dtype=np.float32)).to(device)

    @lazy
    def get_cuda_grid(W, H):
        xs, ys = np.meshgrid(np.arange(0, W, dtype=np.int32), np.arange(0, H, dtype=np.int32))
        return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)

    def fast_cuda_padding(intrinsics):
        eyes = get_cuda_eye(4)
        eyes[:3, :3] = intrinsics
        return eyes

    def fast_cuda_vstack_1(coords):
        ones = get_cuda_ones((coords.shape[0] + 1, *coords.shape[1:]))
        ones[:coords.shape[0]] = coords
        return ones


    class View:
        def __init__(self, idx):
            self.intrinsics, self.extrinsics = read_camera_parameters(
                os.path.join(workdir, 'cams_1/{:0>8}_cam.txt'.format(idx)))
            original_h, original_w = read_size(os.path.join(workdir, 'images/{:0>8}.jpg'.format(idx)))
            self.intrinsics[0] *= img_wh[0] / original_w
            self.intrinsics[1] *= img_wh[1] / original_h
            self.intrinsics_inv = np.linalg.inv(self.intrinsics)
            self.extrinsics_inv = np.linalg.inv(self.extrinsics)
            depth_est = np.squeeze(read_pfm(os.path.join(workdir, 'depth_est/{:0>8}.pfm'.format(idx)))[0], 2)
            self.xyz1 = fast_vstack_1(self.intrinsics_inv @ (get_xy1(depth_est.shape[1], depth_est.shape[0]) * depth_est.reshape(1, -1))).copy()
            self.shape = depth_est.shape

    @lazy
    def get_view(idx):
        return View(idx)

    class CUDAView:
        def __init__(self, idx):
            intrinsics, extrinsics = read_camera_parameters(
                os.path.join(workdir, 'cams_1/{:0>8}_cam.txt'.format(idx)))
            self.intrinsics = torch.from_numpy(intrinsics).to(device)
            self.extrinsics = torch.from_numpy(extrinsics).to(device)
            original_h, original_w = read_size(os.path.join(workdir, 'images/{:0>8}.jpg'.format(idx)))
            self.intrinsics[0] *= img_wh[0] / original_w
            self.intrinsics[1] *= img_wh[1] / original_h
            self.intrinsics_inv = torch.inverse(self.intrinsics)
            self.extrinsics_inv = torch.inverse(self.extrinsics)
            self.depth_path = os.path.join(workdir, 'depth_est/{:0>8}.pfm'.format(idx))
            self.shape = None

        def make(self, compute_depth):
            if compute_depth or self.shape is None:
                depth_est = np.squeeze(read_pfm(self.depth_path, flip=False)[0], 2)
                depth_est_tensor = torch.flip(torch.from_numpy(depth_est).to(device), [0])
                self.xyz1 = fast_cuda_vstack_1(
                    self.intrinsics_inv @ (get_cuda_xy1(depth_est.shape[1], depth_est.shape[0]) * depth_est_tensor.view(1, -1))
                ).clone()
                self.shape = depth_est.shape
                return depth_est_tensor
            return None

    @lazy
    def get_cuda_view(idx):
        return CUDAView(idx)



    # project the reference point cloud into the source view, then project back
    def reproject_with_depth(ref_idx, src_idx, depth_src):
        ## step1. project reference pixels to the source view
        # reference view x, y
        # reference 3D space
        ref_view = get_view(ref_idx)
        src_view = get_view(src_idx)
        # source 3D space
        # source view x, y
        K_xyz_src = ((fast_padding(src_view.intrinsics) @ src_view.extrinsics @ ref_view.extrinsics_inv) @ ref_view.xyz1)[:3]
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        ## step2. reproject the source view points with source view depth estimation
        # find the depth estimation of the source view
        x_src = xy_src[0].reshape(ref_view.shape)
        y_src = xy_src[1].reshape(ref_view.shape)
        assert x_src.dtype == np.float32
        assert y_src.dtype == np.float32
        sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR).reshape(1, -1)
        # mask = sampled_depth_src > 0

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        
        

        xyz1_src = fast_vstack_1(src_view.intrinsics_inv @ (fast_vstack_1(xy_src) * sampled_depth_src))
        # reference 3D space
        xyz_reprojected = ((ref_view.extrinsics @ src_view.extrinsics_inv) @ xyz1_src)[:3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[2].reshape(ref_view.shape)
        K_xyz_reprojected = ref_view.intrinsics @ xyz_reprojected
        xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-6)
        x_reprojected = xy_reprojected[0].reshape(ref_view.shape)
        y_reprojected = xy_reprojected[1].reshape(ref_view.shape)

        assert x_reprojected.dtype == np.float32
        assert y_reprojected.dtype == np.float32

        return depth_reprojected, x_reprojected, y_reprojected


    def reproject_with_depth_by_torch(ref_idx, src_idx, depth_src):
        ref_view = get_cuda_view(ref_idx)
        ref_view.make(False)
        src_view = get_cuda_view(src_idx)
        src_view.make(False)
        K_xyz_src = ((fast_cuda_padding(src_view.intrinsics) @ src_view.extrinsics @ ref_view.extrinsics_inv) @ ref_view.xyz1)[:3]
        xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

        xy_src_cpu = xy_src.cpu().numpy()
        x_src = xy_src_cpu[0].reshape(ref_view.shape)
        y_src = xy_src_cpu[1].reshape(ref_view.shape)
        assert x_src.dtype == np.float32
        assert y_src.dtype == np.float32
        sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR).reshape(1, -1)
        # mask = sampled_depth_src > 0

        # source 3D space
        # NOTE that we should use sampled source-view depth_here to project back
        xyz1_src = fast_cuda_vstack_1(src_view.intrinsics_inv @ (fast_cuda_vstack_1(xy_src) * torch.from_numpy(sampled_depth_src).to(device)))
        # reference 3D space
        xyz_reprojected = ((ref_view.extrinsics @ src_view.extrinsics_inv) @ xyz1_src)[:3]
        # source view x, y, depth
        depth_reprojected = xyz_reprojected[2].view(ref_view.shape)
        K_xyz_reprojected = ref_view.intrinsics @ xyz_reprojected
        xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-6)
        x_reprojected = xy_reprojected[0].view(ref_view.shape)
        y_reprojected = xy_reprojected[1].view(ref_view.shape)

        return depth_reprojected, x_reprojected, y_reprojected

    def check_geometric_consistency(ref_idx, src_idx, depth_ref, depth_src, geo_pixel_thres, geo_depth_thres):
        x_ref, y_ref = get_grid(depth_ref.shape[1], depth_ref.shape[0])
        depth_reprojected, x2d_reprojected, y2d_reprojected = reproject_with_depth(ref_idx, src_idx, depth_src)

        mask = np.logical_and(
            np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2) < geo_pixel_thres,
            (np.abs(depth_reprojected - depth_ref) / depth_ref) < geo_depth_thres
        )
        depth_reprojected[~mask] = 0

        return mask, depth_reprojected

    def check_geometric_consistency_by_torch(ref_idx, src_idx, depth_ref, depth_src, geo_pixel_thres, geo_depth_thres):
        x_ref, y_ref = get_cuda_grid(depth_src.shape[1], depth_src.shape[0])
        depth_reprojected, x2d_reprojected, y2d_reprojected = reproject_with_depth_by_torch(ref_idx, src_idx, depth_src)

        mask = (torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2) < geo_pixel_thres) & ((torch.abs(depth_reprojected - depth_ref) / depth_ref) < geo_depth_thres)
        depth_reprojected[~mask] = 0

        return mask, depth_reprojected


    def filter_depth(fusion_pairs):
        # for the final point cloud
        vertexs = []
        vertex_colors = []

        # for each reference view and the corresponding source views
        for ref_view, src_views in fusion_pairs:
            if color:
                ref_img = read_img(os.path.join(workdir, 'images/{:0>8}.jpg'.format(ref_view)), img_wh)[0]
            # load the estimated depth of the reference view
            if cuda >= 0:
                ref_depth_est = get_cuda_view(ref_view).make(True)
                photo_mask = torch.flip(torch.from_numpy(np.squeeze(read_pfm(os.path.join(workdir, 'confidence/{:0>8}.pfm'.format(ref_view)), False)[0], 2)).to(device), [0]) > photo_thres
            else:
                ref_depth_est = np.squeeze(read_pfm(os.path.join(workdir, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0], 2)
                photo_mask = np.squeeze(read_pfm(os.path.join(workdir, 'confidence/{:0>8}.pfm'.format(ref_view)))[0], 2) > photo_thres

            all_srcview_depth_ests = 0
            # compute the geometric mask
            geo_mask_sum = 0

            for src_view in src_views:

                # the estimated depth of the source view
                src_depth_est = read_pfm(os.path.join(workdir, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]
                
                if cuda >= 0:
                    geo_mask, depth_reprojected = check_geometric_consistency_by_torch(ref_view,
                                                                            src_view,
                                                                            ref_depth_est,
                                                                            src_depth_est,
                                                                            geo_pixel_thres,  geo_depth_thres)
                    geo_mask_sum = geo_mask_sum + geo_mask.int()
                    all_srcview_depth_ests = all_srcview_depth_ests + depth_reprojected
                else:
                    geo_mask, depth_reprojected = check_geometric_consistency(ref_view,
                                                                            src_view,
                                                                            ref_depth_est,
                                                                            src_depth_est,
                                                                            geo_pixel_thres,  geo_depth_thres)
                    geo_mask_sum += geo_mask.astype(np.int32)
                    all_srcview_depth_ests += depth_reprojected

            depth_est_averaged = (all_srcview_depth_ests + ref_depth_est) / (geo_mask_sum + 1)
            geo_mask = geo_mask_sum >= geo_mask_thres

            if cuda >= 0:
                final_mask = photo_mask & geo_mask

                print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(workdir, ref_view, geo_mask.float().mean().item(), photo_mask.float().mean().item(), final_mask.float().mean().item()))

                height, width = depth_est_averaged.shape[:2]
                x, y = get_cuda_grid(width, height)
            else:
                final_mask = np.logical_and(photo_mask, geo_mask)

                # os.makedirs(os.path.join(workdir, "mask"), exist_ok=True)
                # save_mask(os.path.join(workdir, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
                # save_mask(os.path.join(workdir, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
                # save_mask(os.path.join(workdir, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

                print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(workdir, ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

                height, width = depth_est_averaged.shape[:2]
                x, y = get_grid(width, height)

            valid_points = final_mask
            # print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            
            if cuda >= 0:
                view = get_cuda_view(ref_view)

                xyz_ref = view.intrinsics_inv @ (torch.cat((x.view(1, -1), y.view(1, -1), torch.ones_like(x).view(1, -1)), dim=0) * depth)
                xyz_world = (view.extrinsics_inv @ torch.cat((xyz_ref, torch.ones_like(x).float().view(1, -1)), dim=0))[:3]
                vertexs.append(xyz_world.transpose(1, 0).cpu().numpy())

                if color:
                    vertex_colors.append((ref_img[valid_points.cpu().numpy()] * 255).astype(np.uint8))
            else:
                view = get_view(ref_view)

                xyz_ref = view.intrinsics_inv @ (np.vstack((x, y, np.ones_like(x))) * depth)
                xyz_world = (view.extrinsics_inv @ np.vstack((xyz_ref, np.ones_like(x))))[:3]
                vertexs.append(xyz_world.transpose(1, 0))

                if color:
                    vertex_colors.append((ref_img[valid_points] * 255).astype(np.uint8))

            # not use anymore
            del view.xyz1
            
            os.makedirs(os.path.join(workdir, "result"), exist_ok=True)
            if color:
                np.save(os.path.join(workdir, "result", "{:08d}.rgb.npy".format(ref_view)), vertex_colors[-1])
            np.save(os.path.join(workdir, "result", "{:08d}.xyz.npy".format(ref_view)), vertexs[-1])
        
        return vertexs, vertex_colors
        

    pair_path = os.path.join(args.workdir, "pair.txt")
    old_pair_path = os.path.join(args.workdir, "last_pair.txt")
    pair_data = read_pair_file(pair_path)
    old_pair_data = read_pair_file(old_pair_path) if os.path.exists(old_pair_path) and not recompute else []
    estimation_pairs, fusion_pairs = compare_pairs(old_pair_data, pair_data)
    if estimation_pairs != []:
        save_depth(estimation_pairs)
    if fusion_pairs != []:
        xyz, rgb = filter_depth(fusion_pairs)
    else:
        xyz, rgb = [], []
    fusion_pairs_set = { ref_view for ref_view, src_views in fusion_pairs }
    for ref_view, src_views in pair_data:
        if ref_view not in fusion_pairs_set:
            xyz.append(np.load(os.path.join(workdir, "result", "{:08d}.xyz.npy".format(ref_view))))
            if color:
                rgb.append(np.load(os.path.join(workdir, "result", "{:08d}.rgb.npy".format(ref_view))))
    
    print('Total {} points !'.format(sum([v.shape[0] for v in xyz])))
    print("Saving the final model to", output)
    if color:
        write_ply(output, [np.concatenate(xyz, axis=0), np.concatenate(rgb, axis=0)], ['x', 'y', 'z', 'red', 'green', 'blue'])
    else:
        write_ply(output, np.concatenate(xyz, axis=0), ['x', 'y', 'z'])
    with open(pair_path, 'r') as fr:
        with open(old_pair_path, 'w') as fw:
            fw.write(fr.read())


if __name__ == '__main__':
    
    # parse arguments and check
    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
    parser.add_argument('--color', action='store_true', help='color the point cloud')
    parser.add_argument('--cuda', type=int, default=0, help='use cuda to fuse or not (=-1)')
    parser.add_argument('--workdir', '-d', required=True, help='data path')
    parser.add_argument('--output', '-o', required=True, help='path to dump ply model')
    parser.add_argument('--batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--n_views', type=int, default=5, help='num of view')
    parser.add_argument('--img_wh', nargs='+', type=int, default=[800, 600], help='height and width of the image')
    parser.add_argument('--loadckpt', '-l', default='./checkpoints/blendedmvs/model_000015.ckpt', help='load a specific checkpoint')
    parser.add_argument('--iteration', type=int, default=4, help='num of iteration of GRU')
    parser.add_argument('--geo_pixel_thres', '-gp', type=float, default=10, help='pixel threshold for geometric consistency filtering')
    parser.add_argument('--geo_depth_thres', '-gd', type=float, default=0.1, help='depth threshold for geometric consistency filtering')
    parser.add_argument('--geo_mask_thres', '-gm', type=int, default=2, help='mask num threshold for filtering')
    parser.add_argument('--photo_thres', '-pt', type=float, default=0.15, help='threshold for photometric consistency filtering')
    parser.add_argument('--recompute', '-re', action='store_true', help='recompute all')
    args = parser.parse_args()
    if args.cuda >= 0:
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device('cpu')
    print("argv:", sys.argv[1:])
    print_args(args)
    main(workdir=args.workdir,
         batch_size=args.batch_size,
         n_views=args.n_views,
         img_wh=args.img_wh,
         loadckpt=args.loadckpt,
         iteration=args.iteration,
         device=device,
         cuda=args.cuda,
         color=args.color,
         output=args.output,
         photo_thres=args.photo_thres,
         geo_pixel_thres=args.geo_pixel_thres,
         geo_depth_thres=args.geo_depth_thres,
         geo_mask_thres=args.geo_mask_thres,
         recompute=args.recompute)
    