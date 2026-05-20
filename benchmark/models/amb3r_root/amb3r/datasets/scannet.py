import os
import cv2
import numpy as np
import os.path as osp
from collections import deque
from torch.utils.data import Dataset

from dust3r.utils.image import imread_cv2
from .base_many_view_dataset import BaseManyViewDataset
from amb3r.tools.utils import threshold_depth_map
from amb3r.tools.pose_dist import compute_ranking


class Scannet(BaseManyViewDataset):
    def __init__(self, num_seq=1, 
                 test_id=None, full_video=False, 
                 kf_every=1, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.num_seq = num_seq
        self.test_id = test_id
        self.full_video = full_video
        self.kf_every = kf_every

         # load all scenes
        self.load_all_scenes(ROOT)
    
    def __len__(self):
        return len(self.scene_list) * self.num_seq

    def load_all_scenes(self, base_dir):
        
        self.folder = {'train': 'scans', 'val': 'scans', 'test': 'scans_test'}[self.split]
        
        if self.test_id is None:

            self.scene_list = [d for d in os.listdir(osp.join(base_dir, self.folder)) if os.path.isdir(os.path.join(base_dir, self.folder, d))]
        else:
            if isinstance(self.test_id, list):
                self.scene_list = self.test_id
            else:
                self.scene_list = [self.test_id]
                
            print(f"Test_id: {self.test_id}")
    
    def _get_views(self, idx, resolution, num_frames, rng, attempts=0): 

        scene_id = self.scene_list[idx // self.num_seq]

        # Load metadata
        intri_path = osp.join(self.ROOT, self.folder, scene_id, 'intrinsic/intrinsic_depth.txt')
        intri = np.loadtxt(intri_path).astype(np.float32)[:3, :3]

        # Load image data
        image_path = osp.join(self.ROOT, self.folder, scene_id, 'color')
        num_files = len(os.listdir(image_path))


        pose_path = osp.join(self.ROOT, self.folder, scene_id, 'poses.npy')

        if os.path.exists(pose_path):
            pose_all = np.load(pose_path)
        
        else:
            pose_all = []

            for i in range(num_files):
                posepath = osp.join(self.ROOT, self.folder, scene_id, 'pose', f'{i}.txt')
                camera_pose = np.loadtxt(posepath).astype(np.float32)
                pose_all.append(camera_pose)
            pose_all = np.stack(pose_all, axis=0)
            np.save(pose_path, pose_all)
            

        
        # ranking: NxN
        rank_path = osp.join(self.ROOT, self.folder, scene_id, 'ranking.npy')
        if os.path.exists(rank_path):
            ranking = np.load(rank_path)
        else:
            ranking, dist = compute_ranking(pose_all, lambda_t=1.0, normalize=True, batched=True)

            print(ranking.shape)
            np.save(rank_path, ranking[:, :512])
            print(f"Ranking saved to {rank_path}")


        while True:
            ref_idx = np.random.randint(0, num_files)
            k = 256
            topk = ranking[ref_idx][1:k+1]
            replace = False if len(topk) >= num_frames - 1 else True
            support_ids = np.random.choice(topk, size=num_frames - 1, replace=replace)

            imgs_idxs = [ref_idx] + support_ids.tolist()
            poses = pose_all[imgs_idxs]

            if np.isfinite(poses).all():
                break
        
            
        
        imgs_idxs = deque(imgs_idxs)


        views = []

        while len(imgs_idxs) > 0:
            im_idx = imgs_idxs.popleft()

            # Load image data
            impath = osp.join(self.ROOT, self.folder, scene_id, 'color', f'{im_idx}.jpg')
            depthpath = osp.join(self.ROOT, self.folder, scene_id, 'depth', f'{im_idx}.png')

            rgb_image = imread_cv2(impath)
            depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.resize(rgb_image, (depthmap.shape[1], depthmap.shape[0]))


            depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0) / 1000.0   
            # depthmap[depthmap>5.] = 0.0         
            depthmap = threshold_depth_map(
                    depthmap, min_percentile=-1, max_percentile=98
                ).astype(np.float32)
            
            # camera_pose = np.loadtxt(posepath).astype(np.float32)
            camera_pose = pose_all[im_idx].astype(np.float32)



            rgb_ori = rgb_image.copy()
            depth_ori = depthmap.copy()
            intri_ori = intri.copy()
            
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intri, resolution, rng=rng, info=impath)

            # Check if the image is valid
            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0 or (not np.isfinite(camera_pose).all()):
                if self.full_video:
                    print(f"Warning: No valid depthmap found for {impath}")
                    continue
                else:
                    if attempts >= 5:
                        new_idx = rng.integers(0, self.__len__()-1)
                        return self._get_views(new_idx, resolution, num_frames, rng)
                    return self._get_views(idx, resolution, num_frames, rng, attempts+1)
            

            dict_info = dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset='scannet',
                label=osp.join(scene_id, str(im_idx)),
                instance=osp.split(impath)[1],
                is_metric=True,
                orig_img=rgb_ori,
                orig_depthmap=depth_ori,
                orig_camera_intrinsics=intri_ori
            )

                
            views.append(dict_info)
            
       

        
        return views

if __name__ == "__main__":

    num_frames=5
    print('loading dataset')

    dataset = Scannet(split='train', ROOT="./data/scannet", resolution=224, num_seq=1)







    