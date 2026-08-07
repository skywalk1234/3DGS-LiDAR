#----------------------------------------------------------------#
# ReconDrive                                                     #
# Source code: https://github.com/TuojingAI/ReconDrive           #
# Copyright (c) TuojingAI. All rights reserved.                  #
#----------------------------------------------------------------#

import yaml
import argparse
import os
import sys
import subprocess
import torch
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root and models directory to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "models"))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils.train_callback import ExportBestModelCallback, ExportMetricCallback
from dataset.vggt4dgs_data_module import VGGT4DGS_LITDataModule
from models.recondrive_model import ReconDrive_LITModelModule


from utils.snapshot import save_pipeline_snapshot, PIPELINE_DEPLOYMENT

torch.set_float32_matmul_precision('highest')


def load_and_merge_configs(main_cfg_path):
    """Load and merge main config with sub-configs"""
    with open(main_cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return main_cfg


class EpochScheduledCheckpoint(pl.Callback):
    """Save a checkpoint at specific epochs (0-indexed, matching epoch_{epoch:02d} naming).

    E.g. epochs=[20, 40, 60, 80] saves epoch_20.ckpt / epoch_40.ckpt / ... on top of
    last.ckpt / best_module.ckpt when --save_last_only is used.
    """
    def __init__(self, dirpath, epochs, filename='epoch_{epoch:02d}'):
        super().__init__()
        self.dirpath = dirpath
        self.epochs = set(epochs)
        self.filename = filename

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch in self.epochs:
            os.makedirs(self.dirpath, exist_ok=True)
            path = os.path.join(self.dirpath, self.filename.format(epoch=trainer.current_epoch) + '.ckpt')
            trainer.save_checkpoint(path)


def main():
    parser = argparse.ArgumentParser(description='eval argparse')
    parser.add_argument('--cfg_path', type=str, required=True, help='Main config file path')
    parser.add_argument('--pretrained_ckpt', type=str, default='')
    parser.add_argument('--train_4d', action='store_true', help='4dgs')
    parser.add_argument('--devices', type=int, default=None, help='Number of GPUs to use (overrides config)')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Custom checkpoint directory (defaults to <save_dir>/ckpt)')
    parser.add_argument('--save_last_only', action='store_true', help='Only save last.ckpt and best_module.ckpt, skip per-epoch checkpoints (saves disk)')
    parser.add_argument('--overfit', action='store_true', help='Overfit on a single batch (overfit_batches=1, single GPU) to quickly verify training')
    args = parser.parse_args()

    with open(args.cfg_path) as f:
        main_cfg = yaml.load(f, Loader=yaml.FullLoader)

    main_cfg['model_cfg']['batch_size'] = main_cfg['data_cfg']['batch_size']

    # Override devices if specified via command line
    if args.devices is not None:
        main_cfg['devices'] = args.devices
        print(f"Using {args.devices} GPU(s) from command line (overriding config)")
    else:
        print(f"Using {main_cfg['devices']} GPU(s) from config file")

    if args.overfit:
        main_cfg['devices'] = 1
        print("Overfit mode: overfit_batches=1, forced single GPU")

    save_dir = main_cfg['save_dir']

    log_dir = os.path.join(save_dir, 'log')
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    code_dir = os.path.join(save_dir, 'code')
    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    save_pipeline_snapshot(PIPELINE_DEPLOYMENT, code_dir)
    with open(os.path.join(save_dir,'cfg.yaml'),'w') as fw:
        yaml.dump(main_cfg, fw)

    pl.seed_everything(main_cfg['seed'], workers=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='logs'
    )

    if args.train_4d:
        data_module = VGGT4DGS_LITDataModule(
            cfg=main_cfg['data_cfg'],
        )

    if args.pretrained_ckpt:
        litmodel = ReconDrive_LITModelModule(
            cfg=main_cfg['model_cfg'],
            save_dir=log_dir,
            logger=logger
        )
        litmodel.load_pretrained_checkpoint(args.pretrained_ckpt, strict=False, verbose=True)
    else:
        litmodel = ReconDrive_LITModelModule(
            cfg=main_cfg['model_cfg'],
            save_dir=log_dir,
            logger=logger
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best_module',
        save_top_k=1,
        monitor="val/psnr",
        mode="max",
        save_last=True,
        every_n_epochs=1
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch_{epoch:02d}',
        save_top_k=-1,
        every_n_epochs=1,
        save_last=False,
    )

    export_metric_callback = ExportMetricCallback(
        export_dir=log_dir,
        monitor='all',
        best_metric_name='val/psnr',
        best_mode='max',
        start_after_epoch=1,
    )

    callbacks = [checkpoint_callback, LearningRateMonitor(), export_metric_callback]
    if not args.save_last_only:
        callbacks.insert(1, periodic_checkpoint_callback)
    else:
        # save_last_only 模式下，额外在关键 epoch（20/40/60/80）存快照
        callbacks.append(EpochScheduledCheckpoint(ckpt_dir, epochs=[20, 40, 60, 80]))

    trainer = pl.Trainer(
        max_epochs=main_cfg.get('train_epoch', 50),
        accelerator="gpu",
        devices=main_cfg['devices'],  # overfit_batches 模式下只用单卡
        precision="32-true",
        gradient_clip_algorithm="norm",
        # batch_size=1 (forced by OOM), accumulate 2 -> effective batch 2.
        # (Was 8: with only ~30 train samples/epoch that meant only ~4 real
        # parameter updates per epoch, which was too slow to learn.)
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        overfit_batches=1 if args.overfit else 0,
        callbacks=callbacks,
        deterministic=True,
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='ddp_find_unused_parameters_true',
        profiler="simple",
        logger=logger
    )

    torch.use_deterministic_algorithms(mode=True,warn_only=True)
    trainer.fit(litmodel, data_module)

    data_module.setup(stage='test')

    print(f"\nTesting best model...{checkpoint_callback.best_model_path}")
    best_model = ReconDrive_LITModelModule.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(best_model, data_module)

if __name__ == "__main__":
    main()