import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
from gsplat import rasterization
import math
import os

# 1. 读取 .ply 文件
def read_ply_file(ply_file_path):
    ply_data = PlyData.read(ply_file_path)
    vertex_data = ply_data['vertex']

    # 坐标点 (3D位置)
    points = np.stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ], axis=1)

    # 法向量
    normals = np.stack([
        vertex_data['nx'],
        vertex_data['ny'],
        vertex_data['nz']
    ], axis=1)

    # 颜色特征 (SH系数)
    f_dc = np.stack([
        vertex_data['f_dc_0'],
        vertex_data['f_dc_1'],
        vertex_data['f_dc_2']
    ], axis=1)

    # 高阶SH系数 (45维)
    f_rest = np.stack([
        vertex_data[f'f_rest_{i}'] for i in range(45)
    ], axis=1)

    # 其他属性
    attributes = {
        'points': points,
        'normals': normals,
        'f_dc': f_dc,
        'f_rest': f_rest,
        'opacity': vertex_data['opacity'],
        'scale': np.stack([
            vertex_data['scale_0'],
            vertex_data['scale_1'],
            vertex_data['scale_2']
        ], axis=1),
        'rotation': np.stack([
            vertex_data['rot_0'],
            vertex_data['rot_1'],
            vertex_data['rot_2'],
            vertex_data['rot_3']
        ], axis=1)
    }

    return points, attributes


def project_covariance_3d_to_2d(cov_3d, view_matrix, proj_matrix):
    """
    将3D高斯点的协方差矩阵投影到2D图像平面上。

    :param cov_3d: 3D高斯点的协方差矩阵 (3x3)
    :param view_matrix: 视图变换矩阵 (4x4)
    :param proj_matrix: 投影矩阵 (4x4)
    :return: 2D高斯点的协方差矩阵 (2x2)
    """
    # 将3D协方差矩阵投影到相机坐标系
    cov_camera = view_matrix[:3, :3] @ cov_3d @ view_matrix[:3, :3].T

    # 将相机坐标系中的协方差矩阵投影到图像平面
    J = proj_matrix[:3, :3]  # 取投影矩阵的前3x3部分
    cov_image = J @ cov_camera @ J.T

    # 提取2D协方差矩阵（忽略深度信息）
    cov_2d = cov_image[:2, :2]

    return cov_2d

def build_rotation_matrix(rotation):
    """
    根据旋转参数构建旋转矩阵。
    :param rotation: 旋转参数，形状为 (4,)，表示四元数 [w, x, y, z]
    :return: 旋转矩阵，形状为 (3, 3)
    """
    w, x, y, z = rotation
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x*x - 2*y*y]
    ])
    return rotation_matrix

def build_scaling_matrix(scale):
    """
    根据缩放参数构建对角缩放矩阵。
    :param scale: 缩放参数，形状为 (3,)，表示 [sx, sy, sz]
    :return: 缩放矩阵，形状为 (3, 3)
    """
    sx, sy, sz = scale
    scaling_matrix = np.diag([sx, sy, sz])
    return scaling_matrix

def build_covariance_matrix(scale, rotation):
    """
    根据缩放参数和旋转参数构建协方差矩阵。
    :param scale: 缩放参数，形状为 (3,)，表示 [sx, sy, sz]
    :param rotation: 旋转参数，形状为 (4,)，表示四元数 [w, x, y, z]
    :return: 协方差矩阵，形状为 (3, 3)
    """
    # 构建旋转矩阵
    R = build_rotation_matrix(rotation)
    # 构建缩放矩阵
    S = build_scaling_matrix(scale)
    # 计算协方差矩阵
    covariance_matrix = R @ S @ S.T @ R.T
    return covariance_matrix

def pixel_in_2dgs(pixel, mean, covariance, threshold=0.1):
    """
    判断像素点是否在2D高斯点的覆盖范围内。
    :param pixel: 像素点坐标 (x_p, y_p)
    :param mean: 2D高斯点的均值向量 N ( μ_x, μ_y)
    :param covariance: 2D高斯点的协方差矩阵 (N, 2, 2)
    :param threshold: 判断阈值
    :return: 像素点是否在2D高斯点的覆盖范围内
    """
    # 计算像素点与高斯点均值的偏差向量
    delta = np.repeat(np.array(pixel)[None], len(mean), axis=0) - mean
    # print(delta.shape)
    delta = delta.reshape(len(mean), 1, 2)


    # 计算协方差矩阵的逆矩阵
    inv_covariance = np.linalg.inv(covariance)
    # 计算概率密度值
    exponet = -0.5 * np.dot(delta, np.dot(inv_covariance, delta.transpose(0, 2, 1)))
    exponet = exponet.squeeze(1)
    det = 2 * np.pi * np.sqrt(np.linalg.det(covariance))
    # prob_density = np.exp(exponet) / det
    for i in range(len(det)):
        prob_density[i] = np.exp(exponet[i]) / det[i]
    # print(prob_density.shape)
    # exponent = -0.5 * np.sum(delta * np.linalg.solve(covariance, delta.T).T, axis=1)
    # prob_density = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(covariance)))
    # 判断是否在覆盖范围内
    return prob_density > threshold

# 计算全局重要性分数
def compute(attributes, train_images, camera_path,beta=0.5):
    # 读取数据
    points = attributes['points']
    opacity = attributes['opacity']
    scale = attributes['scale']
    rotation = attributes['rotation']

    camera = json.load(open(camera_path))
    # 构建3d协方差矩阵
    cov_3d = [build_covariance_matrix(scale[i], rotation[i]) for i in range(len(points))]
    print("获取相机参数和计算2d高斯点的协方差矩阵")
    # for i in tqdm(range(len(camera)), desc="相机参数处理进度"):
    for i in tqdm(range(2), desc="相机参数处理进度"):
        # 计算视图变换矩阵
        R_cw = np.array(camera[i]['rotation'])
        t_cw = -np.array(camera[i]['position'])  # 注意这里是负的
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw
        # 计算内参矩阵
        K = np.eye(3)
        K[0, 0] = camera[i]['fx']
        K[1, 1] = camera[i]['fy']
        K[0, 2] = camera[i]['width'] / 2
        K[1, 2] = camera[i]['height'] / 2
        # 计算投影矩阵
        P = K @ T_cw[:3, :]
        camera[i]['view_matrix'] = T_cw
        camera[i]['proj_matrix'] = P
        # 计算3D点在图像平面上的投影
        projected_points = (P @ np.hstack([points, np.ones((len(points), 1))]).T).T
        projected_points[:, :2] /= projected_points[:, 2:]
        # 计算2D高斯点的协方差矩阵
        cov_2d = [project_covariance_3d_to_2d(cov_3d[j], T_cw, P) for j in range(len(points))]
        camera[i]['cov_2d'] = cov_2d
        camera[i]['projected_points'] = projected_points
    print("计算完成")
    importance_scores = np.zeros(len(attributes['points']))
    # 计算重叠
    width, height = camera[0]['width'], camera[0]['height']
    print(camera[0]['projected_points'].shape)
    # print(camera[0]['cov_2d'])
    # print(camera[0]['projected_points'])
    print("计算重叠")
    # Multi = len(camera)
    Multi = 5
    for j in tqdm(range(100), desc="像素处理进度"):
        # 计算像素点在图像平面上的坐标
        pixel_x = j % width
        pixel_y = j // width
        # 计算当前是第几张图片
        camera_index = j // (width * height)
        # 找到在该像素点处的2DGS
        # gaussians_at_pixel = []
        # 遍历所有的高斯点
        # for k in range(len(points)):
        #     # 计算2d高斯点是否与像素点重叠
        #     if is_pixel_in_2dgs((pixel_x, pixel_y), camera[camera_index]['projected_points'][k][:2], camera[camera_index]['cov_2d'][k]):
        #         gaussians_at_pixel.append(k)
        gaussians_at_pixel = pixel_in_2dgs((pixel_x, pixel_y), camera[camera_index]['projected_points'][...,:2], camera[camera_index]['cov_2d'])
        print(len(gaussians_at_pixel))
        print(gaussians_at_pixel)
        # 对2dgs进行深度排序
        opacity_at_pixel = opacity[gaussians_at_pixel]
        depth_order = np.argsort(opacity_at_pixel)
        # 计算涉及到的2dgs的αk*兀(1-αk)的积，累加到importance_scores中
        if len(depth_order) == 0:
            continue
        else:
            importance_scores[gaussians_at_pixel[depth_order[0]]] += opacity_at_pixel[depth_order[0]]
            for l in range(1, len(depth_order)):
                k = depth_order[l]
                accumulated_opacity = opacity_at_pixel[k] * np.prod(1 - opacity_at_pixel[depth_order[:l-1]])
                importance_scores[gaussians_at_pixel[k]] += accumulated_opacity
    # 计算体积
    Volume = np.sqrt(np.sum(scale ** 2, axis=1))
    # 计算全局重要性分数
    max90_volume = np.percentile(Volume, 90)
    for j in range(len(importance_scores)):
        V_norm = min(max(Volume[j] / max90_volume, 0), 1)
        importance_scores[j] = importance_scores[j] * (V_norm ** beta)
    print(importance_scores)
    return importance_scores


# 3. 剪枝操作
def prune_points(points, importance_scores, threshold=0.5):
    threshold_value = np.percentile(importance_scores, threshold * 100)
    pruned_indices = importance_scores > threshold_value
    return points[pruned_indices]


# 4. 聚类操作
def cluster_points(attributes, num_clusters=10):
    # 对属性进行预处理
    attr = np.concatenate([
        # attributes['points'],
        # attributes['normals'],
        attributes['f_dc'],     # 3维
        attributes['f_rest'],   # 45维
        attributes['opacity'].reshape(-1, 1),    # 1维
        attributes['scale'],    # 3维
        attributes['rotation']  # 4维
    ],axis=1)   # 56维
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(attr)

    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(scaled_features)
    return kmeans.labels_


# 5. 定义神经网络模型
class PositionalEncoding(nn.Module):
    def __init__(self, num_levels=10, input_dim=3):
        super().__init__()
        self.num_levels = num_levels
        self.input_dim = input_dim
        self.freq_bands = 2 ** torch.arange(num_levels).float() * math.pi  # [10]

    def forward(self, x):
        # x: [B, 3]
        out = [x]
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)  # [B, 3 + 2 * 10 * 3] = [B, 63]

class MLP(nn.Module):
    def __init__(self, pe_levels=10, hidden_dim=256, out_dim=56):
        super(MLP, self).__init__()
        self.pe = PositionalEncoding(num_levels=10)
        in_dim = 3 + 3 * 2 * pe_levels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)  # 输出如 SH 系数等
        )

    def forward(self, x):
        x = self.pe(x)
        return self.net(x)


# 6. 拟合神经网络
def fit_neural_network(points, labels, attributes, num_clusters=10, epochs=30, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attr = np.concatenate([
        attributes['f_dc'],  # 3维
        attributes['f_rest'],  # 45维
        attributes['opacity'].reshape(-1, 1),  # 1维
        attributes['scale'],  # 3维
        attributes['rotation']  # 4维
    ], axis=1)  # 56维

    print(f"开始训练 {num_clusters} 个聚类模型...")
    models = []

    for i in range(num_clusters):
        cluster_points = points[labels == i]
        target = attr[labels == i]
        cluster_points = torch.tensor(cluster_points, dtype=torch.float32).to(device)
        target = torch.tensor(target, dtype=torch.float32).to(device)

        model = MLP().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"\n训练聚类 {i + 1}/{num_clusters}, 包含 {len(cluster_points)} 个点")

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx in range(0, len(cluster_points), batch_size):
                batch = cluster_points[batch_idx:batch_idx + batch_size]
                batch_target = target[batch_idx:batch_idx + batch_size]

                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs} - 平均损失: {avg_loss:.4f}")

        models.append(model)
        print(f"聚类 {i + 1} 训练完成")

    print("\n所有聚类模型训练完成")
    return models


# 7. 微调阶段（简化版）
def fine_tune_models(models, points, labels, epochs=5, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, model in enumerate(models):
        cluster_points = points[labels == i]
        cluster_points = torch.tensor(cluster_points, dtype=torch.float32).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            for batch_idx in range(0, len(cluster_points), batch_size):
                batch = cluster_points[batch_idx:batch_idx + batch_size]
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()


# 8. 保存压缩后的模型
def save_model(points, labels, models, output_file_path):
    el = []
    for i, model in enumerate(models):
        cluster_points = points[labels == i]
        vertex = np.array([(p[0], p[1], p[2]) for p in cluster_points],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        torch.save(model, output_file_path+'/'+f'/model_{i}.pth')
        el.append(PlyElement.describe(vertex, f'vertex{i}'))
    PlyData(el).write(output_file_path+'/compressed_ply_file.ply')

def save_model_new(points, labels, models, output_file_path, cluster_num=10):
    el = []
    for i in range(cluster_num):
        cluster_points = points[labels == i]
        vertex = np.array([(p[0], p[1], p[2]) for p in cluster_points],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el.append(PlyElement.describe(vertex, f'vertex{i}'))
    PlyData(el).write(output_file_path+'/compressed_ply_file.ply')
    torch.save(models, output_file_path+'/models.pth')

def decompress_new(model_path):
    # 读取ply文件
    ply_path = model_path+'/compressed_ply_file.ply'
    ply_data = PlyData.read(ply_path)
    outputs = []
    points = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = torch.load(model_path + f'/models.pth')
    # 获取顶点数据
    for i in range(10):
        vertex_data = ply_data[f'vertex{i}']
        point = np.stack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ], axis=1)
        points.append(point)

        point = torch.tensor(point, dtype=torch.float32).to(device)
        # 标签
        label = torch.tensor(np.full(len(point), i), dtype=torch.long).to(device)
        # 进行预测
        output = model(point, label)
        # 转换为numpy数组
        output = output.detach().cpu().numpy()
        outputs.append(output)
    attr = np.concatenate(outputs, axis=0)
    points = np.concatenate(points, axis=0)
    return points, attr

def decompress(model_path):
    # 读取ply文件
    ply_path = model_path+'/compressed_ply_file.ply'
    ply_data = PlyData.read(ply_path)
    outputs = []
    points = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取顶点数据
    for i in range(10):
        vertex_data = ply_data[f'vertex{i}']
        point = np.stack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ], axis=1)
        points.append(point)
        # 加载模型
        model = torch.load(model_path+f'/model_{i}.pth')

        point = torch.tensor(point, dtype=torch.float32).to(device)
        # 进行预测
        output = model(point)
        # 转换为numpy数组
        output = output.detach().cpu().numpy()
        outputs.append(output)
    attr = np.concatenate(outputs, axis=0)
    points = np.concatenate(points, axis=0)
    return points, attr
    # attr = np.concatenate([
    #     # attributes['points'],
    #     # attributes['normals'],
    #     attributes['f_dc'],  # 3维
    #     attributes['f_rest'],  # 45维
    #     attributes['opacity'].reshape(-1, 1),  # 1维
    #     attributes['scale'],  # 3维
    #     attributes['rotation']  # 4维
    # ], axis=1)

def render_test(means, attr, img_path, camera_path = 'cameras.json'):
    from test import load_camera_json, render_image, calculate_psnr

    viewmat, K, width, height = load_camera_json(camera_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opacity = attr[...,48]
    f_dc = attr[...,:3]
    f_rest = attr[...,3:48]
    scales = attr[...,49:52]
    quats = attr[...,52:]
    # sh_degree = 3  # 假设 SH 系数的阶数为 4
    # num_coeffs = (sh_degree + 1) ** 2
    # colors = []
    f_rest_r = f_rest[:, 0::3]
    f_rest_g = f_rest[:, 1::3]
    f_rest_b = f_rest[:, 2::3]

    # 拼接完整 SH 系数（[N, 16]）
    sh_r = np.concatenate([f_dc[:, 0:1], f_rest_r], axis=1)
    sh_g = np.concatenate([f_dc[:, 1:2], f_rest_g], axis=1)
    sh_b = np.concatenate([f_dc[:, 2:3], f_rest_b], axis=1)

    # 最终 SH 系数数组（[N, 16, 3]）
    colors = np.stack([sh_r, sh_g, sh_b], axis=-1)

    means = torch.tensor(means, dtype=torch.float32, device=device)
    scales = torch.tensor(scales, dtype=torch.float32, device=device)
    quats = torch.tensor(quats, dtype=torch.float32, device=device)
    opacities = torch.tensor(opacity, dtype=torch.float32, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)
    colors = torch.clamp(colors, -1, 1)
    # 确保视角矩阵和内参矩阵的形状正确
    viewmats = viewmat.unsqueeze(0)  # [1, 4, 4]，因为 gsplat 期望批量输入
    Ks = K.unsqueeze(0)  # [1, 3, 3]

    # 调整颜色的维度
    colors = colors[None]  # 使其适应 rasterization 输入要求

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
        sh_degree=3,  # 假设 SH 系数的阶数为 3
    )

    # 提取渲染结果
    rendered_image = rendered_image[0]  # [H, W, 3]，移除批量维度
    rendered_image = rendered_image.cpu().numpy()  # 转换为 NumPy 数组

    # 保存渲染的图像
    output_name = 'compressed_rendered_image.png'
    output_path = os.path.join(img_path, output_name)
    import imageio
    imageio.imwrite(output_path, (rendered_image * 255).astype(np.uint8))
    print(f"Rendered image saved to {output_path}")

# 主函数
def compress_ply_file(input_file_path, output_file_path, camera_path="cameras.json"):
    points,attr = read_ply_file(input_file_path)
    # importance_scores = compute(attr, None, camera_path)
    # pruned_points = prune_points(points, importance_scores)
    labels = cluster_points(attr)
    models = fit_neural_network(points, labels, attr)
    # fine_tune_models(models, pruned_points, labels)
    save_model(points, labels, models, output_file_path)
    # 测试解压
    points, attr = decompress(output_file_path)
    # 测试结果
    render_test(points, attr,output_file_path, camera_path)
    # 测试渲染
    torch.cuda.empty_cache()
    # render(input_file_path, 'cameras.json')

def compress_ply_file_new(input_file_path, output_file_path, camera_path="cameras.json"):
    from models import MultiMLP, train_multimlp
    from importance import compute_importance, compute_importance_optimized
    points, attr = read_ply_file(input_file_path)
    #
    # importance_scores = compute_importance_optimized(attr, camera_path)
    # pruned_points = prune_points(points, importance_scores)
    # # 显示剪枝效果
    # print("减掉了", (len(points) - len(pruned_points))/ len(points)*100, "%")

    labels = cluster_points(attr)
    models = train_multimlp(points, labels, attr, output_file_path)
    save_model_new(points, labels, models, output_file_path)
    # 测试解压
    points, attr = decompress_new(output_file_path)
    # 测试结果
    render_test(points, attr, output_file_path, camera_path)
    # 测试渲染
    torch.cuda.empty_cache()

# 示例用法
# input_file_path = 'point_cloud.ply'
# output_file_path = 'compressed_ply_file.ply'
# camera_path = 'cameras.json'
# compress_ply_file(input_file_path, output_file_path, camera_path)

import argparse

parser = argparse.ArgumentParser(description='PLY文件处理工具')
parser.add_argument('--input', required=True, help='输入PLY文件路径')
parser.add_argument('--output', required=True, help='输出文件路径')
parser.add_argument('--camera', required=True, help='相机参数JSON文件路径')
# # model
# parser.add_argument('--model', required=True, help='模型文件路径')
args = parser.parse_args()
input_file_path = args.input
output_file_path = args.output
camera_path = args.camera
compress_ply_file_new(input_file_path, output_file_path, camera_path)

# f_rest shape:  (1915866, 45)
# Rendered image saved to /output/NeuralGS/model/model3/rendered_image.png
# psnr: 18.691975276654755
# psnr: 14.965773291825503
# (base) root@bupt-computational-photography-zxtm8nwasyxa-main:/openbayes/home/NeuralGS# python test.py
# f_rest shape:  (1915866, 45)
# Rendered image saved to /output/NeuralGS/model/model3/rendered_image.png
# compressed psnr: 18.691975276654755
# uncompressed psnr: 14.965773291825503
# (base) root@bupt-computational-photography-zxtm8nwasyxa-main:/openbayes/home/NeuralGS# python main.py --input /openbayes/home/gaussian-splatting/output/666753c0-3/point_cloud/iteration_30000/point_cloud.ply --output ./model/model3 --camera /openbayes/home/gaussian-splatting/output/666753c0-3/cameras.json
