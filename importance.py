import torch
import json
from tqdm import tqdm


def build_rotation_matrix_batch(rotation):
    """rotation: (N, 4) quaternion w, x, y, z"""
    w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
    B = rotation.shape[0]
    rot = torch.empty((B, 3, 3), device=rotation.device)
    rot[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    rot[:, 0, 1] = 2 * x * y - 2 * z * w
    rot[:, 0, 2] = 2 * x * z + 2 * y * w
    rot[:, 1, 0] = 2 * x * y + 2 * z * w
    rot[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    rot[:, 1, 2] = 2 * y * z - 2 * x * w
    rot[:, 2, 0] = 2 * x * z - 2 * y * w
    rot[:, 2, 1] = 2 * y * z + 2 * x * w
    rot[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return rot


def build_covariance_matrix_batch(scale, rotation):
    R = build_rotation_matrix_batch(rotation)
    S = torch.diag_embed(scale)
    cov3d = R @ S @ S.transpose(1, 2) @ R.transpose(1, 2)
    return cov3d


def project_covariance_batch(cov3d, view_matrix, proj_matrix):
    V = view_matrix[:, :3, :3]
    J = proj_matrix[:, :3, :3]
    cov_camera = V @ cov3d @ V.transpose(1, 2)
    cov_image = J @ cov_camera @ J.transpose(1, 2)
    return cov_image[:, :2, :2]

def auto_select_chunk(base_pixel=512, base_point=1024):
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e6  # MB
    used_mem = torch.cuda.memory_reserved(0) / 1e6
    avail = total_mem - used_mem

    factor = min(avail / 2000, 4)  # 留足 buffer，每 2GB 显存扩大一倍，最多 x4
    return int(base_pixel * factor), int(base_point * factor)

def pixel_in_2dgs_parallel(pixels, means, covariances, threshold=0.1, pixel_chunk=16, point_chunk=16):
    pixel_chunk, point_chunk = auto_select_chunk()
    device = means.device
    pixels = torch.tensor(pixels, dtype=torch.float32, device=device)
    M, N = pixels.shape[0], means.shape[0]
    result = torch.zeros((M, N), dtype=torch.bool, device='cpu')

    for p_start in range(0, M, pixel_chunk):
        p_end = min(p_start + pixel_chunk, M)
        pixel_block = pixels[p_start:p_end]  # (B, 2)

        for g_start in range(0, N, point_chunk):
            g_end = min(g_start + point_chunk, N)
            mean_block = means[g_start:g_end]        # (G, 2)
            cov_block = covariances[g_start:g_end]   # (G, 2, 2)

            delta = pixel_block[:, None, :] - mean_block[None, :, :]  # (B, G, 2)
            delta = delta.unsqueeze(-1)  # (B, G, 2, 1)

            inv_cov = torch.linalg.inv(cov_block).unsqueeze(0).expand(p_end - p_start, g_end - g_start, 2, 2)
            exponent = -0.5 * torch.matmul(
                delta.transpose(-2, -1), torch.matmul(inv_cov, delta)
            ).squeeze(-1).squeeze(-1)

            det_cov = torch.linalg.det(cov_block).unsqueeze(0).expand(p_end - p_start, g_end - g_start)
            prob_density = torch.exp(exponent) / (2 * torch.pi * torch.sqrt(det_cov))
            mask_chunk = (prob_density > threshold)
            result[p_start:p_end, g_start:g_end] = mask_chunk

    return result


def compute_importance(attributes, camera_path, beta=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor(attributes['points'], device=device)
    opacity = torch.tensor(attributes['opacity'], device=device)
    scale = torch.tensor(attributes['scale'], device=device)
    rotation = torch.tensor(attributes['rotation'], device=device)

    with open(camera_path) as f:
        camera_data = json.load(f)

    N = points.shape[0]
    cov_3d = build_covariance_matrix_batch(scale, rotation)

    for i in tqdm(range(len(camera_data)), desc="预处理相机"):
        R_cw = torch.tensor(camera_data[i]['rotation'], dtype=torch.float32, device=device)
        t_cw = -torch.tensor(camera_data[i]['position'], dtype=torch.float32, device=device)
        T_cw = torch.eye(4, device=device)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        K = torch.eye(3, device=device)
        K[0, 0] = camera_data[i]['fx']
        K[1, 1] = camera_data[i]['fy']
        K[0, 2] = camera_data[i]['width'] / 2
        K[1, 2] = camera_data[i]['height'] / 2
        P = K @ T_cw[:3, :]

        camera_data[i]['view_matrix'] = T_cw
        camera_data[i]['proj_matrix'] = P

        ones = torch.ones((N, 1), device=device)
        projected = (P @ torch.cat([points, ones], dim=1).T).T
        projected[:, :2] /= projected[:, 2:3]
        cov_2d = project_covariance_batch(cov_3d, T_cw.unsqueeze(0).expand(N, 4, 4), P.unsqueeze(0).expand(N, 3, 4))

        camera_data[i]['projected_points'] = projected
        camera_data[i]['cov_2d'] = cov_2d

    importance_scores = torch.zeros(N, device=device)

    for i in tqdm(range(len(camera_data)), desc="计算重要性"):
        width, height = camera_data[i]['width'], camera_data[i]['height']
        x = torch.arange(0, width, device=device)
        y = torch.arange(0, height, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        img_pixels = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # shape: (W*H, 2)
        projected_2d = camera_data[i]['projected_points'][:, :2]
        cov_2d = camera_data[i]['cov_2d']
        mask = pixel_in_2dgs_parallel(img_pixels, projected_2d, cov_2d)

        for p_idx, covered_ids in enumerate(mask):
            covered = torch.nonzero(covered_ids).squeeze(1)
            if len(covered) == 0:
                continue
            opacities = opacity[covered]
            order = torch.argsort(opacities)
            first = covered[order[0]]
            importance_scores[first] += opacities[order[0]]
            for l in range(1, len(order)):
                k = covered[order[l]]
                acc = opacities[order[l]] * torch.prod(1 - opacities[order[:l]])
                importance_scores[k] += acc

    Volume = torch.sqrt(torch.sum(scale ** 2, dim=1))
    max90_volume = torch.quantile(Volume, 0.9)
    V_norm = torch.clamp(Volume / max90_volume, 0, 1)
    importance_scores *= V_norm ** beta

    return importance_scores.cpu().numpy()

def compute_importance_optimized(attributes, camera_path, beta=0.5, tile_size=64, dist_thresh=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    points = torch.tensor(attributes['points'], device=device)
    opacity = torch.tensor(attributes['opacity'], device=device)
    scale = torch.tensor(attributes['scale'], device=device)
    rotation = torch.tensor(attributes['rotation'], device=device)
    N = points.shape[0]

    with open(camera_path) as f:
        camera_data = json.load(f)

    cov_3d = build_covariance_matrix_batch(scale, rotation)

    for i in tqdm(range(len(camera_data)), desc="预处理相机"):
        R_cw = torch.tensor(camera_data[i]['rotation'], dtype=torch.float32, device=device)
        t_cw = -torch.tensor(camera_data[i]['position'], dtype=torch.float32, device=device)
        T_cw = torch.eye(4, device=device)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        K = torch.eye(3, device=device)
        K[0, 0] = camera_data[i]['fx']
        K[1, 1] = camera_data[i]['fy']
        K[0, 2] = camera_data[i]['width'] / 2
        K[1, 2] = camera_data[i]['height'] / 2
        P = K @ T_cw[:3, :]

        ones = torch.ones((N, 1), device=device)
        projected = (P @ torch.cat([points, ones], dim=1).T).T
        projected[:, :2] /= projected[:, 2:3]

        cov_2d = project_covariance_batch(
            cov_3d,
            T_cw.unsqueeze(0).expand(N, 4, 4),
            P.unsqueeze(0).expand(N, 3, 4)
        )

        camera_data[i]['projected_points'] = projected.cpu()
        camera_data[i]['cov_2d'] = cov_2d.cpu()
        camera_data[i]['proj_matrix'] = P
        camera_data[i]['view_matrix'] = T_cw

    importance_scores = torch.zeros(N, device=device)

    for i in tqdm(range(len(camera_data)), desc="计算重要性"):
        width, height = camera_data[i]['width'], camera_data[i]['height']
        projected_2d = camera_data[i]['projected_points'].to(device)
        projected_2d = projected_2d[:, :2]
        cov_2d = camera_data[i]['cov_2d'].to(device)

        # 当前相机视角图像中心
        center = torch.tensor([width / 2, height / 2], device=device)
        dist = torch.norm(projected_2d - center, dim=1)
        keep_mask = dist < dist_thresh

        if keep_mask.sum() == 0:
            continue

        selected_points = projected_2d[keep_mask]
        selected_covs = cov_2d[keep_mask]
        selected_opacity = opacity[keep_mask]
        selected_ids = keep_mask.nonzero(as_tuple=True)[0]

        for y0 in range(0, height, tile_size):
            for x0 in range(0, width, tile_size):
                x = torch.arange(x0, min(x0+tile_size, width), device=device)
                y = torch.arange(y0, min(y0+tile_size, height), device=device)
                xx, yy = torch.meshgrid(x, y, indexing='xy')
                img_pixels = torch.stack([xx.flatten(), yy.flatten()], dim=1)

                mask = pixel_in_2dgs_parallel(
                    img_pixels,
                    selected_points,
                    selected_covs,
                    threshold=0.1,
                    pixel_chunk=32,
                    point_chunk=64
                )

                for p_idx, covered_ids in enumerate(mask):
                    covered = torch.nonzero(covered_ids).squeeze(1)
                    if len(covered) == 0:
                        continue
                    opacities = selected_opacity[covered]
                    order = torch.argsort(opacities)
                    first = selected_ids[covered[order[0]]]
                    importance_scores[first] += opacities[order[0]]
                    for l in range(1, len(order)):
                        k = selected_ids[covered[order[l]]]
                        acc = opacities[order[l]] * torch.prod(1 - opacities[order[:l]])
                        importance_scores[k] += acc

    Volume = torch.sqrt(torch.sum(scale ** 2, dim=1))
    max90_volume = torch.quantile(Volume, 0.9)
    V_norm = torch.clamp(Volume / max90_volume, 0, 1)
    importance_scores *= V_norm ** beta

    return importance_scores.cpu().numpy()

