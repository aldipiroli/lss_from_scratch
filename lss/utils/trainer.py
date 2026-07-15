import torch
from tqdm import tqdm

from lss.utils.plotters import plot
from lss.utils.trainer_base import TrainerBase


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
        pass

    @torch.no_grad()
    def evaluate_model(self):
        self.model.eval()
        pass