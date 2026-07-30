import torch


def project_pcl_to_pillar(pcl, bev_cfg):
    x_min = bev_cfg["x_min"]
    x_max = bev_cfg["x_max"]
    y_min = bev_cfg["y_min"]
    y_max = bev_cfg["y_max"]
    cell_size = bev_cfg["cell_size"]

    W = int((x_max - x_min) / cell_size)
    H = int((y_max - y_min) / cell_size)

    x_idx = torch.floor((pcl[..., 0] - x_min) / cell_size).long()
    y_idx = torch.floor((pcl[..., 1] - y_min) / cell_size).long()

    valid = (x_idx >= 0) & (x_idx < W) & (y_idx >= 0) & (y_idx < H)
    pcl_quant = torch.stack((x_idx, y_idx, pcl[..., 2]), dim=-1)
    return pcl_quant, valid


def scatter_feautres_to_bev(feats, bev_idx, bev_cfg):
    cell_size = bev_cfg["cell_size"]
    W = int((bev_cfg["x_max"] - bev_cfg["x_min"]) / cell_size)
    H = int((bev_cfg["y_max"] - bev_cfg["y_min"]) / cell_size)
    C = feats.shape[-1]

    bev_idx = bev_idx.long()
    x = bev_idx[..., 0]
    y = bev_idx[..., 1]

    linear_idx = y * W + x  # (N)
    linear_idx = linear_idx.unsqueeze(-1).expand(-1, C)  # (N,C)

    bev = torch.zeros(
        H * W,
        C,
        device=feats.device,
        dtype=feats.dtype,
    )
    bev.scatter_reduce_(
        dim=0,
        index=linear_idx,
        src=feats,
        reduce="sum",
        include_self=False,
    )
    bev = bev.reshape(H, W, C)
    return bev
