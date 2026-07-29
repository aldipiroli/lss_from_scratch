import argparse
import pickle

import numpy as np
from pyquaternion import Quaternion

from lss.utils.misc import get_logger, load_config, make_artifacts_dirs
from lss.dataset import __all_datasets__

import matplotlib.pyplot as plt


def plot_trajectories(trajectories, output_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(0, 0, c="k", marker="x", s=60, label="Ego")
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], "-o", linewidth=1, markersize=2, alpha=0.8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Extracted Trajectories ({len(trajectories)})")
    ax.set_aspect("equal")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def get_ego_pose(nusc, sample):
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return ego_pose


def global_to_ego(point, ego_pose):
    translation = np.array(ego_pose["translation"])
    rotation = Quaternion(ego_pose["rotation"])
    point = point - translation
    point = rotation.inverse.rotate(point)
    return point


def extract_future_trajectory(dataset, sample, horizon):
    nusc = dataset.nusc
    current_pose = get_ego_pose(nusc, sample)
    trajectory = []
    current = sample

    for _ in range(horizon):
        pose = get_ego_pose(nusc, current)
        global_xyz = np.array(pose["translation"])
        local_xyz = global_to_ego(global_xyz, current_pose)
        trajectory.append(local_xyz[:2])
        if current["next"] == "":
            break
        current = nusc.get("sample", current["next"])
    if len(trajectory) != horizon:
        return None

    return np.asarray(trajectory, dtype=np.float32)


def extract_trajectories(dataset, config, logger):
    horizon = config["LSS"].get("trajectory_horizon", 30)
    trajectories = []
    logger.info(f"Extracting trajectories (horizon={horizon})...")

    for i, sample in enumerate(dataset.samples):
        traj = extract_future_trajectory(
            dataset,
            sample,
            horizon,
        )
        if traj is None:
            continue
        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            logger.info(
                f"{i + 1}/{len(dataset.samples)} samples processed "
                f"({len(trajectories)} trajectories)"
            )
    logger.info(f"Finished. Collected {len(trajectories)} trajectories.")
    return trajectories


def run(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    nuscenes_dataset = __all_datasets__[config["DATA"]["dataset"]]
    train_dataset = nuscenes_dataset(cfg=config, mode="train", logger=logger)
    trajectories = extract_trajectories(train_dataset, config, logger)

    plot_trajectories(trajectories, "tmp.png")

    with open(args.output, "wb") as f:
        pickle.dump(trajectories, f)
    logger.info(f"Saved {len(trajectories)} trajectories to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default="config/config.yaml", help="Config path"
    )
    parser.add_argument(
        "--output", type=str, default="data/nuscenes/trajectories_dict.pkl"
    )
    args = parser.parse_args()
    run(args)
