import numpy as np
import open3d as o3d
##############################################################
# % Author: Jianlin Dou
# % Date:1/11/2025
###############################################################
def preprocess_for_pointtr(
    pcd,
    voxel=0.005,
    denoise=True,
):
    if pcd.is_empty():
        raise RuntimeError("empty point cloud")

    # 1) remove NaN values
    P = np.asarray(pcd.points)
    mask = np.isfinite(P).all(axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[mask]))

    # 3) sparsity sampling
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    # 4) denoising
    if denoise and np.asarray(pcd.points).shape[0] > 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 6) normalize
    P = np.asarray(pcd.points)
    if len(P) == 0:
        raise RuntimeError("point cloud after flitering is empty")
    centroid = P.mean(axis=0, keepdims=True)
    P = P - centroid
    scale = np.linalg.norm(P, axis=1).max()
    if scale < 1e-8:
        scale = 1.0
    P = P / scale

    point_partial = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    return point_partial, centroid, scale

def postprocessing_from_pointr(arr, centroid, scale):
    arr = np.asarray(arr, dtype=np.float32)
    arr[:, :3] = arr[:, :3] * scale + centroid
    point_complete = o3d.geometry.PointCloud()
    point_complete.points = o3d.utility.Vector3dVector(arr)
    return point_complete
