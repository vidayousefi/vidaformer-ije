import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from vformer.data.augment import Augmentation
from vformer.data.loader import Div2kDataset
from vformer.misc.utils import get_unique_file, set_seed
from vformer.models.attacks import Attacks
from vformer.models.coders.dense_coder import CSPDenseCoder
from vformer.models.coders.sade import TransCSPDenseCoder
from vformer.models.coders.unet import HybridUnet
from vformer.models.critics import BasicCritic
from vformer.models.trainer import Trainer


def prepare_data(train_path, val_path):
    train_data = Div2kDataset(train_path, Augmentation.train_transform)
    train = DataLoader(train_data, batch_size=1, num_workers=2, shuffle=True)

    validation_data = Div2kDataset(val_path, Augmentation.val_transform)
    validation = DataLoader(validation_data, batch_size=1, num_workers=0, shuffle=False)
    return train, validation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--data_depth", default=6, type=int)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--train_data", default="D:/Datasets/div2k/train/_", type=str)
    parser.add_argument("--val_data", default="D:/Datasets/div2k/val/_", type=str)
    parser.add_argument(
        "--augmentation", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--decay", default=5e-6, type=float)
    parser.add_argument("--dropout", default=0.03, type=float)
    parser.add_argument("--mse", default=1e-3, type=float)
    parser.add_argument(
        "--validation", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--jpeg", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--noise", default=0, type=float)
    parser.add_argument("--crop", default=0, type=int)
    parser.add_argument("--time", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--coordconv", default=True, type=bool)
    parser.add_argument("--transformer", default=True, type=bool)
    parser.add_argument("--encoder_blocks", default=[3, 6, 3], type=list[int])
    parser.add_argument("--decoder_blocks", default=[6, 3], type=list[int])
    parser.add_argument("--base_channels", default=48, type=int)
    parser.add_argument(
        "--inverse_bottleneck", default=False, action=argparse.BooleanOptionalAction
    )
    # parser.add_argument('--resume', type=str)
    # parser.add_argument('--resume-epoch', type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    set_seed(42)

    opt = parse_args()

    results_dir = "results"

    run = get_unique_file("results", 100)
    log_dir = os.path.join(results_dir, run)
    writer_dir = os.path.join(log_dir, "tensorboard")
    net_dir = os.path.join(log_dir, "network")
    sample_dir = os.path.join(log_dir, "samples")

    os.makedirs(log_dir)
    os.mkdir(net_dir)
    os.makedirs(sample_dir)

    Augmentation.calc_transform(opt.augmentation, opt.resolution)
    train, validation = prepare_data(opt.train_data, opt.val_dir)

    attack = Attacks(opt.noise, opt.crop, opt.jpeg)

    log_hyperparams(opt, results_dir, run)

    if not opt.validation:
        trainer = Trainer(
            opt,
            data_depth=opt.data_depth,
            coder=HybridUnet,
            critic=BasicCritic,
            log_dir=log_dir,
            writer_dir=writer_dir,
            net_dir=net_dir,
            sample_dir=sample_dir,
            attack=attack,
            target_mse=opt.mse,
            resolution=opt.resolution,
            encoder_blocks=opt.encoder_blocks,
            decoder_blocks=opt.decoder_blocks,
            base_channels=opt.base_channels,
            inverse_bottleneck=opt.inverse_bottleneck,
            dropout=opt.dropout,
        )
        trainer.fit(opt, train, validation, epochs=opt.epochs)
        torch.save(
            trainer.model.state_dict(),
            os.path.join(results_dir, "latest-model.pth"),
        )
        trainer.save_self(os.path.join(results_dir, f"latest-trainer.bin"))
    else:
        trainer = Trainer(
            opt,
            data_depth=opt.data_depth,
            coder=HybridUnet,
            critic=BasicCritic,
            log_dir=log_dir,
            writer_dir=writer_dir,
            net_dir=net_dir,
            sample_dir=sample_dir,
            attack=attack,
            target_mse=opt.mse,
            resolution=opt.resolution,
            encoder_blocks=opt.encoder_blocks,
            decoder_blocks=opt.decoder_blocks,
            base_channels=opt.base_channels,
            inverse_bottleneck=opt.inverse_bottleneck,
            dropout=opt.dropout,
        )
        trainer.model.load_state_dict(
            torch.load(os.path.join(results_dir, "latest-model.pth"))
        )
        # trainer = Trainer.load(os.path.join(results_dir, f'latest-trainer.bin'),
        #                         os.path.join(results_dir, 'latest-model.pth'))
        # trainer.after_deserialize(log_dir, writer_dir, net_dir, sample_dir, increment_epoch=False)

        trainer.validate(validation)

        if opt.time:
            trainer.time_it(validation)


# def log_hyperparams(args, results_dir, run):
#     param_log_path = os.path.join(results_dir, 'hyperparameters.log')
#     with open(param_log_path, 'a') as hyper_file:
#         dic = vars(args)
#         dic['run'] = run
#         hyper_file.write(json.dumps(dic) + '\n')


def log_hyperparams(args, results_dir, run):
    param_log_path = os.path.join(results_dir, "hyperparameters.log")

    # Convert args to a dictionary and ensure all values are converted to standard Python types
    dic = {
        key: (
            int(value)
            if isinstance(value, np.integer)
            else (
                float(value)
                if isinstance(value, np.floating)
                else bool(value) if isinstance(value, np.bool_) else value
            )
        )
        for key, value in vars(args).items()
    }

    dic["run"] = run

    with open(param_log_path, "a") as hyper_file:
        hyper_file.write(json.dumps(dic) + "\n")


if __name__ == "__main__":
    main()
