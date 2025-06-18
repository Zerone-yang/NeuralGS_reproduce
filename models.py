import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
from tqdm import trange
from gsplat import rasterization
import math
import os
from torch.utils.tensorboard import SummaryWriter

import torch

# 位置编码
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


# 并行化的MLP模型
class MultiMLP(nn.Module):
    def __init__(self, num_clusters, output_dim=56, hidden_dim=256, pe_levels=10):
        super().__init__()
        self.pe = PositionalEncoding(num_levels=10)
        input_dim = 3 + 3 * 2 * pe_levels
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_clusters)
        ])

    def forward(self, x, cluster_ids):
        """
        x: [B, 3]  (位置编码后的坐标)
        cluster_ids: [B] (int64)
        """
        outputs = torch.zeros((x.shape[0], 56), device=x.device)
        x = self.pe(x)
        for cid in torch.unique(cluster_ids):
            mask = (cluster_ids == cid)
            outputs[mask] = self.mlps[cid](x[mask])
        return outputs


def train_multimlp(points, labels, attributes, train_path, num_clusters=10, epochs=3*60000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attr = np.concatenate([
        attributes['f_dc'],            # 3
        attributes['f_rest'],         # 45
        attributes['opacity'].reshape(-1, 1),  # 1
        attributes['scale'],          # 3
        attributes['rotation']        # 4
    ], axis=1)

    model = MultiMLP(num_clusters).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    points = torch.tensor(points, dtype=torch.float32).to(device)
    targets = torch.tensor(attr, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    N = len(points)

    # === 初始化 TensorBoard ===
    log_dir = os.path.join(train_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    last_quarter_start = int(epochs * 0.75)

    # 使用 tqdm 进度条
    pbar = trange(epochs, desc="Training", ncols=100)

    for epoch in pbar:
        model.train()
        perm = torch.randperm(N)
        points_shuffled = points[perm]
        targets_shuffled = targets[perm]
        labels_shuffled = labels[perm]

        optimizer.zero_grad()
        preds = model(points_shuffled, labels_shuffled)
        loss = criterion(preds, targets_shuffled)
        loss.backward()
        optimizer.step()

        total_loss = loss.item()
        writer.add_scalar("Loss/Total", total_loss, epoch)
        if epoch >= last_quarter_start:
            writer.add_scalar("Loss/LastQuarter", total_loss, epoch - last_quarter_start)

        # 更新 tqdm 描述
        pbar.set_postfix(loss=f"{total_loss:.6f}")

    writer.close()
    print("\n训练完成，TensorBoard 日志保存在:", log_dir)
    return model

