import torch
from tqdm import tqdm

from lss.utils import utils as utils
from lss.utils.plotters import plot
from lss.utils.trainer_base import TrainerBase


class Trainer(TrainerBase):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.all_alpha_bar = utils.precompute_alpha_t(config).to(self.device)
        self.scheaduler = utils.get_noise_scheduler(config).to(self.device)

    def train(self):
        self.logger.info("Started training..")
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.config["OPTIM"]["num_epochs"]):
            self.epoch = epoch
            self.train_one_epoch()
            self.evaluate_model()
            self.save_checkpoint(epoch)

    def train_one_epoch(self, n_train_samples=100):
        self.model.train()
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for n_iter, (x0, labels) in pbar:
            labels = labels.to(self.device)
            labels = utils.drop_class_condition(labels, self.config)
            x0 = x0.to(self.device)
            x_noisy, eps_gt, t = utils.get_noisy_image(x0, self.all_alpha_bar)
            eps_gt = eps_gt.to(self.device)
            t = t.to(self.device)
            eps_pred = self.model(x_noisy, t, labels)
            loss, loss_dict = self.loss_fn(eps_pred, eps_gt)
            self.write_dict_to_tb(loss_dict, self.total_iters_train, prefix="train")
            loss.backward()
            self.accumulate_gradients()
            self.total_iters_train += 1
            pbar.set_postfix(
                {
                    "mode": "train",
                    "epoch": f"{self.epoch}/{self.config['OPTIM']['num_epochs']}",
                    "loss": loss.item(),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            self.write_float_to_tb(
                self.optimizer.param_groups[0]["lr"], "train/lr", self.total_iters_train
            )
        pbar.close()

    @torch.no_grad()
    def evaluate_model(self, char=None, plot_to_image=False, fig_name="0_0"):
        self.model.eval()
        T = self.config["MODEL"]["T"]
        xt = torch.randn(self.config["DATA"]["img_size"]).unsqueeze(0).to(self.device)
        label = torch.tensor(self.val_dataset.inv_letter_mapping[char]).to(self.device)
        no_class = torch.full(
            (xt.shape[0],),
            self.config["MODEL"]["num_classes"],
            dtype=torch.long,
            device=self.device,
        )
        for t in tqdm(range(T - 1, -1, -1), desc=f"Sampling image {fig_name}"):
            t_tensor = torch.tensor(t, dtype=torch.long).unsqueeze(0).to(self.device)
            pred_eps_cond = self.model(xt, t_tensor, label)
            pred_eps_uncond = self.model(xt, t_tensor, no_class)
            xt = utils.denoise_image_step(
                xt,
                pred_eps_cond,
                pred_eps_uncond,
                t_tensor,
                self.scheaduler,
                self.all_alpha_bar,
                weight=self.config["MODEL"]["combination_weight"],
            )
            if t % self.config["MISC"]["plot_every_t_steps"] == 0:
                denoised_img = plot(
                    xt[0],
                    save_figure=plot_to_image,
                    output_path=f"{self.config['IMG_OUT_DIR']}/{str(fig_name).zfill(4)}/{str(t).zfill(5)}.png",
                )
                if not plot_to_image:
                    self.write_images_to_tb(
                        denoised_img,
                        t,
                        f"img/{str(self.total_iters_train).zfill(4)}/sample_{fig_name}",
                    )

        self.logger.info(
            f"[Sample {fig_name}] Final xt stats -> t: {t}, mean: {xt.mean()}, min: {xt.min()}, max: {xt.max()}"
        )
