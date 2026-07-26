import torch
from tqdm import tqdm

from lss.utils.trainer_base import TrainerBase
from lss.utils.camera_utils import batch_data, unbatch_data
from lss.dataset.nuscenes_dataset import CAMERAS

CAMERAS = CAMERAS[:2]


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            self.save_checkpoint(epoch)

    def train_one_epoch(self):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (all_images, intrinsics, extrinsics) in pbar:
            images = batch_data(all_images, CAMERAS)
            images = images.cuda()
            feats, dist = self.model(images)
            feats, dist = self.post_process_model_output(feats, dist)

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        pass

    def post_process_model_output(self, feats, dist):
        feats = unbatch_data(
            feats,
            batch_size=self.config["OPTIM"]["batch_size"],
            n_cameras=len(CAMERAS),
        )
        dist = unbatch_data(
            dist,
            batch_size=self.config["OPTIM"]["batch_size"],
            n_cameras=len(CAMERAS),
        )
        return feats, dist

