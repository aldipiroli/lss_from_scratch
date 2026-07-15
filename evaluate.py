import argparse

from lss.dataset import __all_datasets__
from lss.model.loss import CFGLoss
from lss.model.model import UNet
from lss.utils.misc import get_logger, load_config, make_artifacts_dirs
from lss.utils.trainer import Trainer


def train(args):
    config = load_config(args.config)
    config = make_artifacts_dirs(config, log_datetime=True)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model = UNet(config)
    trainer.set_model(model)
    if args.ckpt is not None:
        trainer.load_checkpoint(args.ckpt)

    dataset = __all_datasets__[config["DATA"]["dataset"]]
    train_dataset = dataset(cfg=config, mode="train", logger=logger)
    val_dataset = dataset(cfg=config, mode="val", logger=logger)
    trainer.set_dataset(
        train_dataset,
        val_dataset,
        data_config=config["DATA"],
        val_set_batch_size=1,
        shuffle_valset_once=False,
    )
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(CFGLoss(config, logger))

    max_len = max(len(line) for line in args.string)
    matrix = [
        [(c.upper() if c != " " else "SPACE") for c in line]
        + ["SPACE"] * (max_len - len(line))
        for line in args.string
    ]
    for i in range(len(matrix)):
        for j in range(max_len):
            char = matrix[i][j]
            trainer.evaluate_model(char=char, plot_to_image=True, fig_name=f"r{i}_c{j}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", type=str, default="config/config.yaml", help="Config path"
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--string",
        default=["hello world", "each letter", "is generated", "using diffusion"],
    )
    args = parser.parse_args()
    train(args)
