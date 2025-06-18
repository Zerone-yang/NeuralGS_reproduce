import numpy as np
import torch
from plyfile import PlyData, PlyElement
import os

def decompress_to_ply(model_path, output_ply_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 ply 顶点位置 + 标签
    ply_path = os.path.join(model_path, 'compressed_ply_file.ply')
    ply_data = PlyData.read(ply_path)

    all_points = []
    all_labels = []

    for i in range(10):  # 假设你是 KMeans 分成10类
        vertex_data = ply_data[f'vertex{i}']
        xyz = np.stack([vertex_data['x'], vertex_data['y'], vertex_data['z']], axis=1)
        all_points.append(xyz)
        all_labels.append(np.full(len(xyz), i))

    all_points = np.concatenate(all_points, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 加载模型
    # from models import MultiMLP  # 替换为定义 MultiMLP 的模块名
    # model = MultiMLP(num_clusters=10).to(device)
    # model.load_state_dict(torch.load(os.path.join(model_path, 'models.pth')))
    # model.eval()
    model = torch.load(os.path.join(model_path,'models.pth')).to(device)

    with torch.no_grad():
        points_tensor = torch.tensor(all_points, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long).to(device)
        outputs = model(points_tensor, labels_tensor).cpu().numpy()

    # 拆解输出
    f_dc = outputs[:, 0:3]
    f_rest = outputs[:, 3:48]
    opacity = outputs[:, 48]
    scale = outputs[:, 49:52]
    rot = outputs[:, 52:56]

    scale = np.exp(scale)
    # 单位化 rotation
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    # clip opacity（如果模型输出不稳定）
    opacity = np.clip(opacity, 0.0, 1.0)

    # 构建输出 ply 顶点字段
    vertex_data = []
    for i in range(all_points.shape[0]):
        row = (
            *all_points[i],       # x, y, z
            *[0, 0, 0],            # nx,ny,nz
            *f_dc[i],             # f_dc_0~2
            *f_rest[i],            # f_rest_0~44
            opacity[i],           # opacity
            *scale[i],          # scale_0~2
            *rot[i],             # rot_0~3
        )
        vertex_data.append(row)

    # 构建 ply 字段描述
    property_names = [
                         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                         ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
                     ] + [(f'f_rest_{i}', 'f4') for i in range(45)] + [  # SH 阶数=3
                         ('opacity', 'f4'), ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
                         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
                     ]
    dtype = np.dtype(property_names)
    vertex_array = np.array(vertex_data, dtype=dtype)

    # 写入 ply 文件
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el], text=False).write(output_ply_path)
    print(f"[✓] 解压后的 ply 文件已保存到: {output_ply_path}")


if __name__ == '__main__':
    model_path = './model/model4'
    output_ply_path = './model/model4/decompressed_ply_file.ply'
    decompress_to_ply(model_path, output_ply_path)