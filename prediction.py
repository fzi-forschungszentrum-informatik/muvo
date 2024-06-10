import git
import os
import socket
import time
from weakref import proxy

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import ModelSummary, LearningRateMonitor

from muvo.config import get_parser, get_cfg
from muvo.data.dataset import DataModule
from muvo.trainer import WorldModelTrainer

from clearml import Task, Dataset, Model


class SaveGitDiffHashCallback(pl.Callback):
    def setup(self, trainer, pl_model, stage):
        repo = git.Repo()
        trainer.git_hash = repo.head.object.hexsha
        trainer.git_diff = repo.git.diff(repo.head.commit.tree)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['world_size'] = trainer.world_size
        checkpoint['git_hash'] = trainer.git_hash
        checkpoint['git_diff'] = trainer.git_diff


class MyModelCheckpoint(ModelCheckpoint):
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        filename = filepath.split('/')[-1]
        _checkpoint = trainer._checkpoint_connector.dump_checkpoint(self.save_weights_only)
        try:
            torch.save(_checkpoint, filename)
        except AttributeError as err:
            key = "hyper_parameters"
            _checkpoint.pop(key, None)
            print(f"Warning, `{key}` dropped from checkpoint. An attribute is not picklable: {err}")
            torch.save(_checkpoint, filename)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    # task = Task.init(project_name=cfg.CML_PROJECT, task_name=cfg.CML_TASK, task_type=cfg.CML_TYPE, tags=cfg.TAG)
    # task.connect(cfg)
    # cml_logger = task.get_logger()
    #
    # dataset_root = Dataset.get(dataset_project=cfg.CML_PROJECT,
    #                            dataset_name=cfg.CML_DATASET,
    #                            ).get_local_copy()

    # data = DataModule(cfg, dataset_root=dataset_root)
    data = DataModule(cfg)

    input_model = Model(model_id='').get_local_copy() if cfg.PRETRAINED.CML_MODEL else None
    # input_model = cfg.PRETRAINED.PATH
    model = WorldModelTrainer(cfg.convert_to_dict(), pretrained_path=input_model)
    # model = WorldModelTrainer.load_from_checkpoint(checkpoint_path=input_model)
    # model.get_cml_logger(cml_logger)

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir)

    callbacks = [
        ModelSummary(),
        SaveGitDiffHashCallback(),
        LearningRateMonitor(),
        MyModelCheckpoint(
            save_dir, every_n_train_steps=cfg.VAL_CHECK_INTERVAL,
        ),
    ]

    if cfg.LIMIT_VAL_BATCHES in [0, 1]:
        limit_val_batches = float(cfg.LIMIT_VAL_BATCHES)
    else:
        limit_val_batches = cfg.LIMIT_VAL_BATCHES

    replace_sampler_ddp = not cfg.SAMPLER.ENABLED

    trainer = pl.Trainer(
        # devices=cfg.GPUS,
        accelerator='auto',
        # strategy='ddp',
        precision=cfg.PRECISION,
        # sync_batchnorm=True,
        max_epochs=None,
        max_steps=cfg.STEPS,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.LOGGING_INTERVAL,
        val_check_interval=cfg.VAL_CHECK_INTERVAL * cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        check_val_every_n_epoch=None,
        # limit_val_batches=limit_val_batches,
        limit_val_batches=3,
        # use_distributed_sampler=replace_sampler_ddp,
        accumulate_grad_batches=cfg.OPTIMIZER.ACCUMULATE_GRAD_BATCHES,
        num_sanity_val_steps=0,
        profiler='simple',
    )

    # trainer.fit(model, datamodule=data)
    trainer.test(model, dataloaders=data.test_dataloader())


if __name__ == '__main__':
    main()
            

if __name__ == '__main__':
    main()
