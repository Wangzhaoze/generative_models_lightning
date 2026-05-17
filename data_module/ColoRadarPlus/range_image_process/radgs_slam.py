import os
os.environ.setdefault("MPLBACKEND", "Agg")

from data_module.ColoRadarPlus.range_image_process.radgs_dataloader import Lidar, ColoRadarPlusDataset, Record
from data_module.ColoRadarPlus.range_image_process.radgs_utils import numpy_to_open3d_pointcloud, merge_point_clouds, visualize_point_cloud, compute_range_image_with_visibility
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange

class LidarSLAM:
    def __init__(self, dataset: ColoRadarPlusDataset):
        self.dataset = dataset
        self.dataset.lidar = self.sync(dataset.ego, dataset.lidar)
        self.dataset.lidar.interpolate_poses(dataset.ego)



    def sync(self, src_rec: Record, tgt_rec: Record) -> Record:
        # Synchronize the source and target records
        src_time_stamps = src_rec.time_stamps
        tgt_time_stamps = tgt_rec.time_stamps

        # Find the common time stamps
        common_time_stamps = np.intersect1d(src_time_stamps, tgt_time_stamps)

        # Synchronize the source and target records
        src_indices = np.searchsorted(src_time_stamps, common_time_stamps)
        tgt_indices = np.searchsorted(tgt_time_stamps, common_time_stamps)

        # Create a new synchronized record
        synced_tgt = type(tgt_rec)(
            files=[tgt_rec.files[i] for i in tgt_indices],
            time_stamps=common_time_stamps,
            calib_path=getattr(tgt_rec, "calib_path", None)
        )

        return synced_tgt
    
    def generate_scene_pcd(
        self,
        distance_max: float = 100.0,
        distance_min: float = 0.05,
        max_frames: int | None = None,
        visualize_scene: bool = False,
        visualize_range_images: bool = False,
    ) -> o3d.geometry.PointCloud:
        """
        input: self.dataset.lidar pointcloudes
        output: scene_pcd
        """
        scene_pcd: list = []
        total_frames = len(self.dataset.lidar)
        if max_frames is not None:
            total_frames = min(total_frames, max(0, int(max_frames)))

        for idxTimeStamp in trange(0, total_frames, 1, desc=f"Generating scene point cloud for {self.dataset.rec_id}"):
            frame_pcd_array = self.dataset.lidar[idxTimeStamp][:, 0:3]

            distances = np.linalg.norm(frame_pcd_array, axis=1)

            if distance_max is not None:
                mask = (distances < distance_max) & (distances > distance_min)
                frame_pcd_array = frame_pcd_array[mask]
                
            frame_pcd = numpy_to_open3d_pointcloud(frame_pcd_array)

            T_global2ego = self.dataset.lidar.lidar_calib_poses[idxTimeStamp]

            global_pcd = frame_pcd.transform(T_global2ego)

            scene_pcd.append(global_pcd)

        if scene_pcd:
            scene_pcd = merge_point_clouds(scene_pcd)
        else:
            scene_pcd = o3d.geometry.PointCloud()

        if visualize_range_images:
            self.visualize_range_images()

        if visualize_scene:
            self.visualize_scene_point_cloud(scene_pcd)

        return scene_pcd

    def visualize_scene_point_cloud(
        self,
        scene_pcd: o3d.geometry.PointCloud,
        title: str = "LiDAR SLAM Scene Reconstruction",
    ) -> None:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        visualize_point_cloud([scene_pcd, axis], title=title)

    def visualize_range_images(self, num_samples: int = 6,
                               az_fov: float = 180, el_fov: float = 60,
                               az_res: float = 1, el_res: float = 1):
        n = len(self.dataset.lidar)
        indices = np.linspace(0, n - 1, num_samples, dtype=int)

        fig, axes = plt.subplots(num_samples, 1, figsize=(16, 2 * num_samples))
        fig.suptitle("LiDAR Range Images (sampled frames)", fontsize=12)

        for row, idx in enumerate(indices):
            pts = self.dataset.lidar[idx][:, :3]
            # compute range image in lidar-local frame (identity pose)
            range_img, _, _ = compute_range_image_with_visibility(
                xyz_abs=pts,
                antenna_pose=np.eye(4),
                az_fov=az_fov,
                el_fov=el_fov,
                az_res=az_res,
                el_res=el_res,
            )
            ax = axes[row] if num_samples > 1 else axes
            ax.imshow(range_img, cmap='jet', aspect='auto',
                      vmin=0, vmax=np.percentile(range_img[range_img > 0], 95) if range_img.any() else 1)
            ax.set_title(f"Frame {idx}", fontsize=9)
            ax.axis('off')

        plt.tight_layout()

        backend = matplotlib.get_backend().lower()
        if "agg" in backend:
            save_dir = os.path.join("..", "..", "output", "range_image_process")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.dataset.rec_id}_range_images.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[Info] Non-interactive backend ({backend}), saved range image preview to: {save_path}")
            plt.close(fig)
        else:
            plt.show()
    

# class RGBDSLAM:
#     def __init__(self, dataset: ColoRadarPlusDataset):
#         self.dataset = dataset
#         self.dataset.depth = self.sync(dataset.ego, dataset.depth)
#         self.dataset.depth.interpolate_poses(dataset.ego)


#     def sync(self, src_rec: Record, tgt_rec: Record) -> Record:
#         # Synchronize the source and target records
#         src_time_stamps = src_rec.time_stamps
#         tgt_time_stamps = tgt_rec.time_stamps

#         # Find the common time stamps
#         common_time_stamps = np.intersect1d(src_time_stamps, tgt_time_stamps)

#         # Synchronize the source and target records
#         src_indices = np.searchsorted(src_time_stamps, common_time_stamps)
#         tgt_indices = np.searchsorted(tgt_time_stamps, common_time_stamps)

#         tgt_indices = np.array([
#         np.argmin(np.abs(tgt_time_stamps - t))
#         for t in src_time_stamps
#         ])

#         # Create a new synchronized record
#         synced_tgt = type(tgt_rec)(
#             files=[tgt_rec.files[i] for i in tgt_indices],
#             time_stamps=[tgt_rec.time_stamps[i] for i in tgt_indices],
#             calib_path=getattr(tgt_rec, "calib_path", None)
#         )


#         return synced_tgt
    
#     def generate_scene_pcd(self):
#         scene_pcd: list = []
#         # range_threshold = self.radar.radar_config.max_det_range
#         distance_max = 100
#         distance_min = 0.05

#         for idxTimeStamp in trange(0, 100, 1, desc=f"Generating scene point cloud for {self.dataset.rec_id}"): 
#             frame_depth = self.dataset.depth[idxTimeStamp]

#             frame_pcd_array = self._depth2pcd(frame_depth)

#             distances = np.linalg.norm(frame_pcd_array, axis=1)

#             if distance_max is not None:
#                 mask = (distances < distance_max) & (distances > distance_min)
#                 frame_pcd_array = frame_pcd_array[mask]

#             T_global2ego = self.dataset.depth.depth_calib_poses[idxTimeStamp]

#             frame_pcd = numpy_to_open3d_pointcloud(frame_pcd_array)

#             global_pcd = frame_pcd.transform(T_global2ego)

#             scene_pcd.append(global_pcd)
        
#         scene_pcd = merge_point_clouds(scene_pcd)
#         visualize_point_cloud(scene_pcd)

#         # filter noise points by clustering
#         scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
#         visualize_point_cloud(scene_pcd)
        
#         return np.asarray(scene_pcd.points)
    
#     def _depth2pcd(self, depth: np.ndarray):
#         fx, fy = 425.7078, 425.7078
#         cx, cy = 422.8032, 238.6957

#         h, w = depth.shape

#         # === 构建像素网格 ===
#         u, v = np.meshgrid(np.arange(w), np.arange(h))

#         # === 反投影到3D空间 ===
#         z = depth
#         x = (u - cx) * z / fx
#         y = (v - cy) * z / fy

#         # 组合成点云 (N,3)
#         points = np.stack((x, y, z), axis=-1)
#         points = points.reshape(-1, 3)

#         # 去除无效点（深度为0的）
#         valid = (z > 0).reshape(-1)
#         points = points[valid]

#         return points


if __name__ == "__main__":
    dataset = ColoRadarPlusDataset()
    slam = LidarSLAM(dataset)

    slam.generate_scene_pcd(visualize_scene=True, visualize_range_images=True)


    # dataset = ColoRadarPlusDataset()
    # slam = RGBDSLAM(dataset)
    # slam.generate_scene_pcd()
