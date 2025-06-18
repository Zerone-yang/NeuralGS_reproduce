import torch
import numpy as np
import json
import trimesh
import matplotlib.pyplot as plt
import gsplat

device = "cuda"

def load_gaussians_from_ply(ply_path):
    import plyfile
    from plyfile import PlyData

    # 用 plyfile 读取，兼容性更好
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex'].data

    def get_props(*names):
        return torch.tensor(np.stack([vertex[name] for name in names], axis=-1), dtype=torch.float32, device=device)

    xyz = get_props("x", "y", "z")
    colors = get_props("f_dc_0", "f_dc_1", "f_dc_2")
    scales = get_props("scale_0", "scale_1", "scale_2")
    rotations = get_props("rot_0", "rot_1", "rot_2", "rot_3")
    opacity = torch.tensor(vertex["opacity"], dtype=torch.float32, device=device)

    return {
        'xyz': xyz,
        'colors': colors,
        'scales': scales,
        'rotations': rotations,
        'opacity': opacity
    }


def load_camera(camera_json_path):
    with open(camera_json_path, 'r') as f:
        cam = json.load(f)

    # 获取内参
    fx = cam["fx"]
    fy = cam["fy"]
    cx = cam["width"] / 2
    cy = cam["height"] / 2
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0, 1]
    ], dtype=torch.float32, device=device)

    # 外参
    R = torch.tensor(cam["rotation"], dtype=torch.float32)  # 3x3
    T = torch.tensor(cam["position"], dtype=torch.float32).reshape(3, 1)  # 3x1
    cam2world = torch.eye(4, dtype=torch.float32)
    cam2world[:3, :3] = R
    cam2world[:3, 3:] = T
    world2cam = torch.inverse(cam2world).to(device)

    return {
        "intrinsics": intrinsics,
        "world2cam": world2cam,
        "width": cam["width"],
        "height": cam["height"]
    }

def render(gaussians, camera, bg_color=(1.0, 1.0, 1.0)):
    cam = gsplat.Camera(camera['world2cam'], camera['intrinsics'], camera['width'], camera['height'])

    raster_settings = gsplat.RasterizationSettings(
        image_height=camera['height'],
        image_width=camera['width'],
        bg=torch.tensor(bg_color, dtype=torch.float32, device=device)
    )

    image = gsplat.rasterize_gaussians(
        cam,
        means3D=gaussians['xyz'],
        scales=gaussians['scales'],
        rotations=gaussians['rotations'],
        colors_precomp=gaussians['colors'],
        opacities=gaussians['opacity'],
        raster_settings=raster_settings
    )
    return image.clamp(0, 1)

if __name__ == "__main__":
    ply_path = '/output/gaussian-splatting/output/666753c0-3/point_cloud/iteration_30000/point_cloud.ply'
    camera_json_path = '/output/gaussian-splatting/output/666753c0-3/cameras.json'

    gaussians = load_gaussians_from_ply(ply_path)
    camera = load_camera(camera_json_path)

    image = render(gaussians, camera)

    # 显示
    plt.imshow(image.cpu().numpy())
    plt.axis('off')
    plt.title("Rendered from 3DGS")
    plt.show()
