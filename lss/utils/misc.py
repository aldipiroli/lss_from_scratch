import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def to_device(x):
    x = x.to(get_device())
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_artifacts_dirs(lss, log_datetime=False):
    if log_datetime:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        now = "logs/"
    lss_artifacts = Path(lss["ARTIFACTS_DIR"])
    lss_artifacts = lss_artifacts / now
    lss["IMG_OUT_DIR"] = lss_artifacts / "imgs"
    os.makedirs(lss["IMG_OUT_DIR"], exist_ok=True)

    lss["LOG_DIR"] = lss_artifacts / "logs"
    os.makedirs(lss["LOG_DIR"], exist_ok=True)

    lss["TB_LOG_DIR"] = lss_artifacts / "tb_logs"
    os.makedirs(lss["TB_LOG_DIR"], exist_ok=True)

    lss["CKPT_DIR"] = lss_artifacts / "ckpts"
    os.makedirs(lss["CKPT_DIR"], exist_ok=True)
    return lss


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.log")
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
