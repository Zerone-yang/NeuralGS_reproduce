import json
import torch
from plyfile import PlyData, PlyElement
from gsplat import rasterization


def load_camera_json(json_path):
    with open(json_path, 'r') as f:
        camera_data = json.load(f)

    # 选择第一个视角
    first_camera = camera_data[0]

    # 提取宽度和高度
    width = int(first_camera['width'])
    height = int(first_camera['height'])

    # 提取旋转矩阵和平移向量
    rotation = np.array(first_camera['rotation'], dtype=np.float32)  # [3, 3]
    position = np.array(first_camera['position'], dtype=np.float32)  # [3]

    # 提取内参参数
    fx = float(first_camera['fx'])
    fy = float(first_camera['fy'])
    # 假设主点在图像中心
    cx = width / 2.0
    cy = height / 2.0

    # 构造内参矩阵 K
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    # 构造视角矩阵（View Matrix）
    R = rotation  # [3, 3]
    R_T = R.T  # 转置，[3, 3]
    t = position  # [3]
    t = -np.dot(R_T, t)  # [3]

    # 构建 4x4 视角矩阵
    viewmat = np.eye(4, dtype=np.float32)  # [4, 4]
    viewmat[:3, :3] = R_T  # 填入旋转部分
    viewmat[:3, 3] = t  # 填入平移部分

    # 转换为 PyTorch 张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viewmat = torch.tensor(viewmat, dtype=torch.float32, device=device)
    K = torch.tensor(K, dtype=torch.float32, device=device)

    return viewmat, K, width, height


# 2. 读取 model.ply 文件
def load_ply_file(ply_path):
    plydata = PlyData.read(ply_path)
    vertex_data = plydata['vertex']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 提取 3D 高斯参数
    means = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T  # [N, 3]
    scales = np.vstack((vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2'])).T  # [N, 3]
    quats = np.vstack(
        (vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3'])).T  # [N, 4]
    opacities = np.array(vertex_data['opacity'], dtype=np.float32)  # [N]

    # sh_degree = 4  # 假设 SH 系数的阶数为 4
    # num_coeffs = (sh_degree + 1) ** 2
    # colors = []

    f_dc = np.stack((vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']), axis=1)
    num_sh_order = 3
    num_rest_coeffs = (num_sh_order + 1) ** 2 - 1
    f_rest = np.stack([vertex_data[f'f_rest_{i}'] for i in range(num_rest_coeffs*3)], axis=1)
    print("f_rest shape: ", f_rest.shape)
    # 将 f_rest 拆成 RGB 三个通道
    f_rest_r = f_rest[:, 0::3]
    f_rest_g = f_rest[:, 1::3]
    f_rest_b = f_rest[:, 2::3]

    # 拼接完整 SH 系数（[N, 16]）
    sh_r = np.concatenate([f_dc[:, 0:1], f_rest_r], axis=1)
    sh_g = np.concatenate([f_dc[:, 1:2], f_rest_g], axis=1)
    sh_b = np.concatenate([f_dc[:, 2:3], f_rest_b], axis=1)

    # 最终 SH 系数数组（[N, 16, 3]）
    colors = np.stack([sh_r, sh_g, sh_b], axis=-1)
    # # 低阶 SH 系数
    # colors.append(np.vstack((vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2'])).T)
    #
    # # 高阶 SH 系数
    # for i in range(num_coeffs - 1):
    #     field_name = f'f_rest_{i}'
    #     colors.append(np.vstack((
    #         vertex_data[field_name],
    #         vertex_data[field_name],
    #         vertex_data[field_name]
    #     )).T)

    # 将颜色列表堆叠为一个数组
    # colors = np.vstack(colors)  # [N, K, 3]

    # 转换为 PyTorch 张量
    means = torch.tensor(means, dtype=torch.float32, device=device)
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    quats = torch.tensor(quats, dtype=torch.float32, device=device)
    opacities = torch.tensor(opacities, dtype=torch.float32, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    colors = torch.clamp(colors, -1.0, 1.0)

    # 打印形状以确认
    print("SH min/max", colors.min().item(), colors.max().item())
    print("opacity range:", opacities.min().item(), opacities.max().item())
    print("scale mean:", scales.mean(dim=0))

    return means, quats, scales, opacities, colors


# 3. 渲染图像
def render_image(ply_path, camera_json_path, output_path="rendered_image.png"):
    # 加载相机参数
    viewmat, K, width, height = load_camera_json(camera_json_path)

    # 加载 PLY 文件
    means, quats, scales, opacities, colors = load_ply_file(ply_path)

    # 确保视角矩阵和内参矩阵的形状正确
    viewmats = viewmat.unsqueeze(0)  # [1, 4, 4]，因为 gsplat 期望批量输入
    Ks = K.unsqueeze(0)  # [1, 3, 3]

    # 调整颜色的维度
    colors = colors[None]   # 使其适应 rasterization 输入要求

    # 调用 gsplat 的光栅化函数
    rendered_image, rendered_depth, _ = rasterization(
        means=means,  # [N, 3]
        quats=quats,  # [N, 4]
        scales=torch.exp(scales),  # [N, 3]
        opacities=torch.sigmoid(opacities),  # [N]
        colors=colors,  # [N, K, 3] 或 [C, N, K, 3]
        viewmats=viewmats,  # [1, 4, 4]
        Ks=Ks,  # [1, 3, 3]
        width=width,
        height=height,
        render_mode="RGB",  # 渲染 RGB 图像
        sh_degree=3,  # 假设 SH 系数的阶数为 4
    )

    # 提取渲染结果
    rendered_image = rendered_image[0]  # [H, W, 3]，移除批量维度
    # rendered_image = torch.clamp(rendered_image, 0.0, 1.0) ** (1.0 / 2.2)
    rendered_image = rendered_image.cpu().numpy()  # 转换为 NumPy 数组

    # 保存渲染的图像
    import imageio
    imageio.imwrite(output_path, (rendered_image * 255).astype(np.uint8))
    print(f"Rendered image saved to {output_path}")


import numpy as np
import cv2


def calculate_psnr(image1_path, image2_path):
    # 读取图像
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # 将图像转换为灰度图像（如果是彩色图像）
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算均方误差（MSE）
    mse = np.mean((img1_gray - img2_gray) ** 2)

    # 如果 MSE 为 0，意味着两张图像完全相同，PSNR 为无限大
    if mse == 0:
        return float('inf')

    # 计算最大像素值（对于 8 位图像，MAX_I = 255）
    max_pixel = 255.0

    # 计算 PSNR
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr


import cv2
# import numpy as np
from math import log10
from PIL import Image

def compute_psnr(img_path1, img_path2):
    # 读取图片，并转换为 RGB
    img1 = np.array(Image.open(img_path1).convert("RGB"), dtype=np.float32)
    img2 = np.array(Image.open(img_path2).convert("RGB"), dtype=np.float32)

    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 完全相同

    PIXEL_MAX = 255.0
    psnr = 10 * log10((PIXEL_MAX ** 2) / mse)
    return psnr

from plyfile import PlyData

def infer_sh_degree(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # 获取所有字段名
    field_names = vertex.data.dtype.names

    # 统计 f_dc_* 和 f_rest_* 字段数量
    f_dc_fields = [f for f in field_names if f.startswith("f_dc_")]
    f_rest_fields = [f for f in field_names if f.startswith("f_rest_")]

    num_dc = len(f_dc_fields)  # 通常应为 3
    num_rest = len(f_rest_fields)

    # 总系数数 = DC (1) + REST (X)，三通道
    num_coeffs_per_channel = 1 + (num_rest // 3)

    # 反推 SH 阶数：num_coeffs = (l+1)^2 → l = sqrt(num_coeffs) - 1
    import math
    sh_degree = int(math.sqrt(num_coeffs_per_channel) - 1)

    print(f"Inferred SH degree: {sh_degree}")
    print(f"Total SH coeffs per channel: {num_coeffs_per_channel}")
    return sh_degree

# 用法
# infer_sh_degree('/output/gaussian-splatting/output/666753c0-3/point_cloud/iteration_30000/point_cloud.ply')


# 4. 执行渲染
if __name__ == "__main__":
    # ply_path = "point_cloud.ply"  # 替换为你的 model.ply 文件路径
    # camera_json_path = "cameras.json"  # 替换为你的 camera.json 文件路径
    # output_path = "rendered_image.png"  # 输出图像路径
    # render_image(ply_path, camera_json_path, output_path)
    # print("psnr:", compute_psnr("compressed_rendered_image.png", "./frames/frame_0001.jpg"))
    # print("psnr:", compute_psnr("rendered_image.png", "./frames/frame_0001.jpg"))

    render_image('/output/gaussian-splatting/output/666753c0-3/point_cloud/iteration_30000/point_cloud.ply',
                 '/output/gaussian-splatting/output/666753c0-3/cameras.json',
                 '/output/NeuralGS/model/model4/rendered_image.png')
    # print("compressed psnr:", compute_psnr("/output/NeuralGS/model/model4/00000 [12].png", "/output/NeuralGS/model/model3/DSC05572.jpg"))
    print("uncompressed psnr:", compute_psnr("/output/NeuralGS/model/model4/rendered_image.png", "/output/NeuralGS/model/model3/DSC05572.jpg"))
    # print("com vs uncom", compute_psnr("/output/NeuralGS/model/model4/rendered_image.png", "/output/NeuralGS/model/model4/compressed_rendered_image.png"))
    # import numpy as np
    # import cv2

    # img1 = cv2.imread("/output/NeuralGS/model/model4/00000.png").astype(np.float32)
    # img2 = cv2.imread("/output/NeuralGS/model/model3/DSC05572.jpg").astype(np.float32)
    # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    #
    # diff = np.abs(img1 - img2).astype(np.uint8)
    # cv2.imwrite("/output/NeuralGS/model/model4/error_map.png", diff)