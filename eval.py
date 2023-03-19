import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import time
if __name__ == '__main__':
    from datasets import MVSDataset
    from models import Pipeline
    import utils
else:
    from .datasets import MVSDataset
    from .models import Pipeline
    from . import utils
import sys
import cv2

cudnn.benchmark = True

def main(workdir,
         batch_size,
         n_views,
         img_wh,
         loadckpt,
         iteration,
         device,
         cuda,
         use_color,
         output,
         dump_depth,
         photo_thres,
         geo_pixel_thres,
         geo_depth_thres,
         geo_mask_thres,
         recompute,
         redirect,
         base_dataset=None
    ):

    if base_dataset is None:
        test_dataset = MVSDataset(workdir, n_views, img_wh)
    else:
        test_dataset = base_dataset

    with open(redirect, 'a') as stream:

        def lazy(func):
            same_dict = {}
            def wrapper(*arglst):
                key = ';'.join([str(arg) for arg in arglst])
                if key not in same_dict:
                    same_dict[key] = func(*arglst)
                return same_dict[key]
            return wrapper

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
        def save_depth():
            if dump_depth:
                os.makedirs(dump_depth, exist_ok=True)
            # dataloader
            TestImgLoader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4, drop_last=False)

            # model
            model = Pipeline(iteration=iteration, test=True)
            model = nn.DataParallel(model)
            model.cuda()

            # load checkpoint file specified by args.loadckpt
            stream.write("loading model {}\n".format(loadckpt))
            state_dict = torch.load(loadckpt)
            model.load_state_dict(state_dict['model'])
            model.eval()
            
            with torch.no_grad():
                for batch_idx, sample in enumerate(TestImgLoader):
                    start_time = time.time()
                    sample_cuda = utils.tocuda(sample)
                    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"],
                                sample_cuda["depth_min"], sample_cuda["depth_max"])

                    outputs = utils.tensor2numpy(outputs)
                    del sample_cuda
                    stream.write('Iter {}/{}, time = {:.3f}\n'.format(batch_idx, len(TestImgLoader), time.time() - start_time))

                    # save depth maps and confidence maps
                    for view_id, depth_est, confidence in zip(sample["view_id"], outputs["depths_upsampled"], outputs["confidence_upsampled"]):
                        view_id = view_id.item()
                        depth_est = np.squeeze(depth_est, 0)
                        test_dataset.view_data[view_id].depth[0] = depth_est
                        test_dataset.view_data[view_id].confidence[0] = np.squeeze(confidence, 0)
                        if dump_depth:
                            np.save(os.path.join(dump_depth, '{:08d}.npy'.format(view_id)), depth_est)

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
                view_data = test_dataset.view_data[idx]
                self.intrinsics = view_data.intrinsics
                self.extrinsics = view_data.extrinsics
                self.intrinsics_inv = np.linalg.inv(self.intrinsics)
                self.extrinsics_inv = np.linalg.inv(self.extrinsics)
                depth_est = view_data.depth[0]
                self.xyz1 = fast_vstack_1(self.intrinsics_inv @ (get_xy1(depth_est.shape[1], depth_est.shape[0]) * depth_est.reshape(1, -1))).copy()
                self.shape = depth_est.shape

        @lazy
        def get_view(idx):
            return View(idx)

        class CUDAView:
            def __init__(self, idx):
                view_data = test_dataset.view_data[idx]
                self.intrinsics = torch.from_numpy(view_data.intrinsics).to(device)
                self.extrinsics = torch.from_numpy(view_data.extrinsics).to(device)
                self.intrinsics_inv = torch.inverse(self.intrinsics)
                self.extrinsics_inv = torch.inverse(self.extrinsics)
                self.depth_est = view_data.depth[0]
                self.shape = None

            def make(self, compute_depth):
                if compute_depth or self.shape is None:
                    depth_est_tensor = torch.from_numpy(self.depth_est).to(device)
                    self.xyz1 = fast_cuda_vstack_1(
                        self.intrinsics_inv @ (get_cuda_xy1(self.depth_est.shape[1], self.depth_est.shape[0]) * depth_est_tensor.view(1, -1))
                    ).clone()
                    self.shape = self.depth_est.shape
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
                if use_color:
                    ref_img = test_dataset.view_data[ref_view].LOD['level_0']
                # load the estimated depth of the reference view
                if cuda >= 0:
                    ref_depth_est = get_cuda_view(ref_view).make(True)
                    confidence = torch.from_numpy(test_dataset.view_data[ref_view].confidence[0]).to(device)
                    photo_mask = confidence > photo_thres
                else:
                    ref_depth_est = test_dataset.view_data[ref_view].depth[0]
                    confidence = test_dataset.view_data[ref_view].confidence[0]
                    photo_mask = confidence > photo_thres

                all_srcview_depth_ests = 0
                # compute the geometric mask
                geo_mask_sum = 0

                for src_view in src_views:

                    # the estimated depth of the source view
                    src_depth_est = test_dataset.view_data[src_view].depth[0]
                    
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

                    stream.write("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}\n".format(workdir, ref_view, geo_mask.float().mean().item(), photo_mask.float().mean().item(), final_mask.float().mean().item()))

                    height, width = depth_est_averaged.shape[:2]
                    x, y = get_cuda_grid(width, height)
                else:
                    final_mask = np.logical_and(photo_mask, geo_mask)

                    stream.write("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}\n".format(workdir, ref_view, geo_mask.mean(), photo_mask.mean(), final_mask.mean()))

                    height, width = depth_est_averaged.shape[:2]
                    x, y = get_grid(width, height)

                valid_points = final_mask
                x, y, depth, confidence = x[valid_points], y[valid_points], depth_est_averaged[valid_points], confidence[valid_points]
                
                if cuda >= 0:
                    view = get_cuda_view(ref_view)

                    xyz_ref = view.intrinsics_inv @ (torch.cat((x.view(1, -1), y.view(1, -1), torch.ones_like(x).view(1, -1)), dim=0) * depth)
                    xyz_world = (view.extrinsics_inv @ torch.cat((xyz_ref, torch.ones_like(x).float().view(1, -1)), dim=0))[:3].transpose(1, 0).cpu().numpy()
                    confidence = confidence.cpu().numpy()

                    if use_color:
                        vertex_color = ref_img[valid_points.cpu().numpy()]
                else:
                    view = get_view(ref_view)

                    xyz_ref = view.intrinsics_inv @ (np.vstack((x, y, np.ones_like(x))) * depth)
                    xyz_world = (view.extrinsics_inv @ np.vstack((xyz_ref, np.ones_like(x))))[:3].transpose(1, 0)

                    if use_color:
                        vertex_color = ref_img[valid_points]

                # not use anymore
                del view.xyz1
                
                if base_dataset is None:
                    os.makedirs(os.path.join(workdir, "result"), exist_ok=True)
                    if use_color:
                        np.save(os.path.join(workdir, "result", "{:08d}.rgb.npy".format(ref_view)), vertex_color)
                    np.save(os.path.join(workdir, "result", "{:08d}.xyz.npy".format(ref_view)), xyz_world)
                else:
                    point_data = test_dataset.view_data[ref_view].points
                    point_data.xyz[0] = xyz_world
                    if use_color:
                        point_data.rgb[0] = vertex_color
                    point_data.conf[0] = confidence

                if output:
                    vertexs.append(xyz_world)
                    if use_color:
                        vertex_colors.append(vertex_color)

            
            return vertexs, vertex_colors
            

        pair_path = os.path.join(workdir, "pair.txt")
        old_pair_path = os.path.join(workdir, "last_pair.txt")
        pair_data = read_pair_file(pair_path)
        old_pair_data = read_pair_file(old_pair_path) if os.path.exists(old_pair_path) and not recompute else []
        estimation_pairs, fusion_pairs = utils.compare_pairs(old_pair_data, pair_data)
        if estimation_pairs != []:
            test_dataset.update(estimation_pairs)
            save_depth()
        if fusion_pairs != []:
            xyz, rgb = filter_depth(fusion_pairs)
        else:
            xyz, rgb = [], []

        if output:
            fusion_pairs_set = { ref_view for ref_view, src_views in fusion_pairs }
            for ref_view, src_views in pair_data:
                if ref_view not in fusion_pairs_set:
                    xyz.append(np.load(os.path.join(workdir, "result", "{:08d}.xyz.npy".format(ref_view))))
                    if use_color:
                        rgb.append(np.load(os.path.join(workdir, "result", "{:08d}.rgb.npy".format(ref_view))))
        
            stream.write('Total {} points !\n'.format(sum([v.shape[0] for v in xyz])))
            stream.write("Saving the final model to " + output + '\n')
            if use_color:
                utils.write_ply(output, [np.concatenate(xyz, axis=0), (np.concatenate(rgb, axis=0) * 127.5 + 127.5).astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
            else:
                utils.write_ply(output, np.concatenate(xyz, axis=0), ['x', 'y', 'z'])
        with open(pair_path, 'r') as fr:
            with open(old_pair_path, 'w') as fw:
                fw.write(fr.read())

    # exit stream
    


if __name__ == '__main__':
    
    # parse arguments and check
    parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
    parser.add_argument('--color', action='store_true', help='color the point cloud')
    parser.add_argument('--cuda', type=int, default=0, help='use cuda to fuse or not (=-1)')
    parser.add_argument('--workdir', '-d', required=True, help='data path')
    parser.add_argument('--output', '-o', default='', help='path to dump ply model')
    parser.add_argument('--dump_depth', default='', help='folder to dump estimated depth')
    parser.add_argument('--batch_size', type=int, default=8, help='testing batch size')
    parser.add_argument('--n_views', type=int, default=5, help='num of view')
    parser.add_argument('--img_wh', nargs='+', type=int, default=[640, 480], help='height and width of the image')
    parser.add_argument('--loadckpt', '-l', default='./checkpoints/blendedmvs/model_000015.ckpt', help='load a specific checkpoint')
    parser.add_argument('--iteration', type=int, default=4, help='num of iteration of GRU')
    parser.add_argument('--geo_pixel_thres', '-gp', type=float, default=10, help='pixel threshold for geometric consistency filtering')
    parser.add_argument('--geo_depth_thres', '-gd', type=float, default=0.1, help='depth threshold for geometric consistency filtering')
    parser.add_argument('--geo_mask_thres', '-gm', type=int, default=2, help='mask num threshold for filtering')
    parser.add_argument('--photo_thres', '-pt', type=float, default=0.15, help='threshold for photometric consistency filtering')
    parser.add_argument('--recompute', '-re', action='store_true', help='recompute all')
    parser.add_argument('--redirect', '-rd', type=str, default='/dev/stdout', help='redirect stdout')
    args = parser.parse_args()
    if args.cuda >= 0:
        device = torch.device('cuda:{}'.format(args.cuda))
    else:
        device = torch.device('cpu')
    with open(args.redirect, 'a') as stream:
        stream.write('argv: ' + str(sys.argv[1:]) + '\n')
        utils.print_args(args, stream)
    main(
        workdir=args.workdir,
        batch_size=args.batch_size,
        n_views=args.n_views,
        img_wh=args.img_wh,
        loadckpt=args.loadckpt,
        iteration=args.iteration,
        device=device,
        cuda=args.cuda,
        use_color=args.color,
        output=args.output,
        dump_depth=args.dump_depth,
        photo_thres=args.photo_thres,
        geo_pixel_thres=args.geo_pixel_thres,
        geo_depth_thres=args.geo_depth_thres,
        geo_mask_thres=args.geo_mask_thres,
        recompute=args.recompute,
        redirect=args.redirect
    )
    