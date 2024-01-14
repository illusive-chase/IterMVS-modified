from .eval import main as eval_main
from typing import Callable, Tuple, Any
import numpy as np
from pathlib import Path
from .datasets import MVSDataset, read_img, ViewData, PointsData
import torch

def main(
    workdir: str,
    view_range: Any,
    image_path_fn: Callable[[int], str],
    extrinsic_fn: Callable[[int], np.ndarray],
    intrinsic_fn: Callable[[int], np.ndarray],
    depth_range_fn: Callable[[int], Tuple[float, float]],
    cuda: int
):
    class PseudoDataset(MVSDataset):
        def __init__(self):
            super().__init__(workdir, n_views=5, img_wh=(640,640), use_prior=False)

        def load(self, view_id):
            if view_id not in self.view_data:
                intrinsics = intrinsic_fn(view_id).astype(np.float32)
                extrinsics = extrinsic_fn(view_id).astype(np.float32)
                depth_min, depth_max = depth_range_fn(view_id)
                img_path = image_path_fn(view_id)
                LOD, original_h, original_w = read_img(img_path, self.img_wh[1], self.img_wh[0])
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
                    prior=None,
                    depth=[None],
                    confidence=[None],
                    LOD=LOD,
                    points=PointsData(
                        rgb=[None],
                        xyz=[None],
                        feature=[None],
                        conf=[None],
                        depth=[None],
                        mask=[None]
                    )
                )

    view_range = np.array(list(view_range))
    with open(Path(workdir) / "pair.txt", "w") as f:
        f.write(f"{view_range.shape[0]}\n")
        origins = np.array([np.linalg.inv(extrinsic_fn(ref_view))[:3, 3] for ref_view in view_range]) # Nx3
        distances = ((origins[None, :, :] - origins[:, None, :]) ** 2).sum(2) # NxN
        for ref_view, nearest_neighbors in zip(view_range, np.argsort(distances, axis=1)[:, 1:5]):
            f.write(f"{ref_view}\n4 ")
            f.write("{} 0 {} 0 {} 0 {} 0\n".format(*view_range[nearest_neighbors]))

    eval_main(
        workdir=workdir,
        batch_size=8,
        n_views=5,
        img_wh=(640,640),
        loadckpt=None,
        iteration=4,
        device=torch.device(f'cuda:{cuda}') if cuda >= 0 else torch.device('cpu'),
        cuda=cuda,
        use_color=True,
        output=str(Path(workdir) / 'output.ply'),
        dump_depth=False,
        photo_thres=0.15,
        geo_pixel_thres=10,
        geo_depth_thres=0.1,
        geo_mask_thres=2,
        recompute=True,
        redirect='/dev/stdout',
        base_dataset=PseudoDataset()
    )
    